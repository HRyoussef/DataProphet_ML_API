# retrain.py
import argparse
import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn

from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ─────────────────────────────────────────────
# 0. Configuration
# ─────────────────────────────────────────────
TRACKING_URI  = "http://localhost:5000"
EXPERIMENT    = "dataprophet-tree-retrain"
MODEL_NAME    = "DataProphet"
DATA_PATH     = "data.pkl"
HELP_DATA_DIR = "help_data/"

# Seuil : on promeut en Staging si le nouveau RMSE
# est inférieur au RMSE Production de plus de ce pourcentage
IMPROVEMENT_THRESHOLD = 0.02   # 2% d'amélioration minimum

TARGET   = "anneedeplantation"
FEATURES = ["genre_bota", "espece", "stadededeveloppement",
            "hauteurarbre", "typenature", "latitude", "longitude"]

numeric_features     = ["latitude", "longitude"]
categorical_features = ["genre_bota", "espece", "stadededeveloppement",
                        "typenature", "hauteurarbre"]

# ─────────────────────────────────────────────
# 1. Connexion MLflow
# ─────────────────────────────────────────────
mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment(EXPERIMENT)
client = MlflowClient(tracking_uri=TRACKING_URI)

# ─────────────────────────────────────────────
# 2. Chargement des données de base
# ─────────────────────────────────────────────
print("📂 Chargement des données de base...")
df = pd.read_pickle(DATA_PATH)
df = df[FEATURES + [TARGET]].dropna()
print(f"   {len(df)} lignes chargées depuis {DATA_PATH}")

# ─────────────────────────────────────────────
# 3. Fusion avec les données de feedback (help_data/)
# ─────────────────────────────────────────────
os.makedirs(HELP_DATA_DIR, exist_ok=True)
feedback_files = [
    f for f in os.listdir(HELP_DATA_DIR)
    if f.endswith(".csv")
]

if feedback_files:
    print(f"📬 {len(feedback_files)} fichier(s) de feedback trouvé(s) :")
    dfs_feedback = []
    for fname in feedback_files:
        path = os.path.join(HELP_DATA_DIR, fname)
        df_fb = pd.read_csv(path)
        df_fb = df_fb[FEATURES + [TARGET]].dropna()
        dfs_feedback.append(df_fb)
        print(f"   + {fname} ({len(df_fb)} lignes)")

    df = pd.concat([df] + dfs_feedback, ignore_index=True)
    print(f"   → Dataset final : {len(df)} lignes")
else:
    print("📭 Aucune donnée de feedback (help_data/ vide) — entraînement sur données de base uniquement")

# ─────────────────────────────────────────────
# 4. Préparation des données
# ─────────────────────────────────────────────
X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ─────────────────────────────────────────────
# 5. Définition du pipeline
# ─────────────────────────────────────────────
preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(),                              numeric_features),
    ("cat", OrdinalEncoder(handle_unknown="use_encoded_value",
                           unknown_value=-1),              categorical_features),
])

pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(
        n_estimators = 200,
        random_state = 42,
        n_jobs       = -1,
    ))
])

# ─────────────────────────────────────────────
# 6. Récupération du RMSE du modèle en Production
# ─────────────────────────────────────────────
production_rmse = None

versions = client.search_model_versions(f"name='{MODEL_NAME}'")
for v in versions:
    if v.current_stage == "Production":
        run = client.get_run(v.run_id)
        production_rmse = run.data.metrics.get("RMSE")
        print(f"\n🏭 Modèle en Production : version {v.version} — RMSE = {production_rmse:.4f}")
        break

if production_rmse is None:
    print("\n⚠️  Aucun modèle en Production trouvé — la promotion sera faite sans comparaison")

# ─────────────────────────────────────────────
# 7. Entraînement + instrumentation MLflow
# ─────────────────────────────────────────────
print("\n🏋️  Entraînement en cours...")

with mlflow.start_run() as run:

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)

    # Params
    mlflow.log_param("n_estimators",    200)
    mlflow.log_param("random_state",    42)
    mlflow.log_param("feedback_files",  len(feedback_files))
    mlflow.log_param("total_rows",      len(df))

    # Métriques
    mlflow.log_metric("MAE",  mae)
    mlflow.log_metric("RMSE", rmse)
    mlflow.log_metric("R2",   r2)

    # Modèle → Registry
    mlflow.sklearn.log_model(
        sk_model              = pipeline,
        name                  = "model",
        registered_model_name = MODEL_NAME,
    )

    new_version = client.search_model_versions(f"name='{MODEL_NAME}'")
    new_version = max(new_version, key=lambda v: int(v.version)).version

    print(f"\n✅ Run terminé — version {new_version} enregistrée")
    print(f"   MAE  : {mae:.4f}")
    print(f"   RMSE : {rmse:.4f}")
    print(f"   R²   : {r2:.4f}")

# ─────────────────────────────────────────────
# 8. Comparaison et promotion automatique en Staging
# ─────────────────────────────────────────────
print("\n🔍 Comparaison avec le modèle en Production...")

if production_rmse is None:
    # Pas de modèle en Production → on promeut directement
    promote = True
    print("   Aucun modèle en Production → promotion automatique")
else:
    improvement = (production_rmse - rmse) / production_rmse
    print(f"   RMSE Production : {production_rmse:.4f}")
    print(f"   RMSE Nouveau    : {rmse:.4f}")
    print(f"   Amélioration    : {improvement*100:.2f}% (seuil : {IMPROVEMENT_THRESHOLD*100}%)")
    promote = improvement >= IMPROVEMENT_THRESHOLD

if promote:
    client.transition_model_version_stage(
        name    = MODEL_NAME,
        version = new_version,
        stage   = "Staging",
    )
    print(f"\n🚀 Version {new_version} promue en Staging automatiquement")
    print(f"   → Lance promote_model.py pour la passer en Production après validation")
else:
    print(f"\n⏸️  Version {new_version} laissée en 'None' — amélioration insuffisante")
    print(f"   → Le modèle en Production reste en place")

# ─────────────────────────────────────────────
# 9. État final du Registry
# ─────────────────────────────────────────────
print("\n📋 État final du Registry :\n")
for v in client.search_model_versions(f"name='{MODEL_NAME}'"):
    icons = {"Production": "✅", "Staging": "🔶", "Archived": "📦"}
    icon  = icons.get(v.current_stage, "  ")
    print(f"  {icon} Version {v.version} → {v.current_stage}")
