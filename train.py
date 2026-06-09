# train.py
import argparse
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# ─────────────────────────────────────────────
# 0. Hyperparamètres via arguments CLI
#    → permet de lancer avec des valeurs différentes sans éditer le fichier
# ─────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--n_estimators", type=int, default=100)
#parser.add_argument("--max_depth",    type=int, default=None)
def nullable_int(val):
    return None if val == "None" else int(val)

parser.add_argument("--max_depth", type=nullable_int, default=None)
parser.add_argument("--min_samples_split", type=int, default=2)
args = parser.parse_args()

# ─────────────────────────────────────────────
# 1. Connexion au serveur MLflow (celui dans Docker)
# ─────────────────────────────────────────────
mlflow.set_tracking_uri("http://localhost:5000")
import os
os.makedirs("mlflow_data/artifacts", exist_ok=True)
mlflow.set_experiment("dataprophet-tree-prediction")

# ─────────────────────────────────────────────
# 2. Chargement des données
# ─────────────────────────────────────────────
df = pd.read_pickle("data.pkl")  

TARGET   = "anneedeplantation"
FEATURES = ["genre_bota", "espece", "stadededeveloppement",
            "hauteurarbre", "typenature", "latitude", "longitude"]

df = df[FEATURES + [TARGET]].dropna()

X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ─────────────────────────────────────────────
# 3. Définition du pipeline
# ─────────────────────────────────────────────
numeric_features     = ["latitude", "longitude"]
categorical_features = ["genre_bota", "espece", "stadededeveloppement", "typenature", "hauteurarbre"]

preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(),                        numeric_features),
    ("cat", OrdinalEncoder(handle_unknown="use_encoded_value",
                           unknown_value=-1),        categorical_features),
])

pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(
        n_estimators      = args.n_estimators,
        max_depth         = args.max_depth,
        min_samples_split = args.min_samples_split,
        random_state      = 42,
        n_jobs            = -1,
    ))
])

# ─────────────────────────────────────────────
# 4. Entraînement + Instrumentation MLflow
# ─────────────────────────────────────────────
with mlflow.start_run():

    # — Entraînement —
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    # — Métriques —
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)

    # — Log des hyperparamètres —
    mlflow.log_param("n_estimators",       args.n_estimators)
    mlflow.log_param("max_depth",          args.max_depth)
    mlflow.log_param("min_samples_split",  args.min_samples_split)

    # — Log des métriques —
    mlflow.log_metric("MAE",  mae)
    mlflow.log_metric("RMSE", rmse)
    mlflow.log_metric("R2",   r2)

    # — Log du modèle + enregistrement dans le Registry —
    mlflow.sklearn.log_model(
        sk_model        = pipeline,
        name            = "model",
        registered_model_name = "DataProphet",   # ← crée/incrémente dans le Registry
    )

    print(f"✅ Run terminé")
    print(f"   MAE  : {mae:.2f}")
    print(f"   RMSE : {rmse:.2f}")
    print(f"   R²   : {r2:.4f}")
    print(f"   Params: n_estimators={args.n_estimators}, "
          f"max_depth={args.max_depth}, min_samples_split={args.min_samples_split}")
