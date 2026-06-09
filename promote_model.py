# promote_model.py
from mlflow.tracking import MlflowClient

# ─────────────────────────────────────────────
# 1. Connexion au serveur
# ─────────────────────────────────────────────
client = MlflowClient(tracking_uri="http://localhost:5000")

MODEL_NAME = "DataProphet"

# ─────────────────────────────────────────────
# 2. Lister toutes les versions du modèle
# ─────────────────────────────────────────────
print(f"\n📋 Versions disponibles pour '{MODEL_NAME}' :\n")

versions = client.search_model_versions(f"name='{MODEL_NAME}'")

for v in versions:
    print(f"  Version {v.version}")
    print(f"    Stage   : {v.current_stage}")
    print(f"    Run ID  : {v.run_id}")
    print(f"    Status  : {v.status}")
    print()

# ─────────────────────────────────────────────
# 3. Récupérer la meilleure version via les métriques
# ─────────────────────────────────────────────
best_version = None
best_rmse    = float("inf")

for v in versions:
    run = client.get_run(v.run_id)
    rmse = run.data.metrics.get("RMSE", float("inf"))
    print(f"  Version {v.version} → RMSE = {rmse:.4f}")

    if rmse < best_rmse:
        best_rmse    = rmse
        best_version = v.version

print(f"\n🏆 Meilleure version : {best_version} (RMSE = {best_rmse:.4f})")

# ─────────────────────────────────────────────
# 4. Archiver l'ancienne version en Production (si elle existe)
# ─────────────────────────────────────────────
for v in versions:
    if v.current_stage == "Production":
        print(f"\n📦 Archivage de l'ancienne version {v.version} (Production → Archived)")
        client.transition_model_version_stage(
            name    = MODEL_NAME,
            version = v.version,
            stage   = "Archived",
        )

# ─────────────────────────────────────────────
# 5. Promouvoir la meilleure version en Production
# ─────────────────────────────────────────────
client.transition_model_version_stage(
    name    = MODEL_NAME,
    version = best_version,
    stage   = "Production",
)

print(f"✅ Version {best_version} promue en Production")

# ─────────────────────────────────────────────
# 6. Confirmation finale
# ─────────────────────────────────────────────
print(f"\n📋 État final du Registry :\n")
for v in client.search_model_versions(f"name='{MODEL_NAME}'"):
    icon = "✅" if v.current_stage == "Production" else "  "
    print(f"  {icon} Version {v.version} → {v.current_stage}")
