import os
from datetime import datetime

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator

MLFLOW_TRACKING_URI = "http://mlflow:5000"
MODEL_NAME = "DataProphet"
IMPROVEMENT_THRESHOLD = 0.01  # 1% d'amélioration minimum sur le RMSE


def evaluate_metrics(**context):
    import mlflow
    from mlflow.tracking import MlflowClient

    os.environ["MLFLOW_SERVER_ALLOWED_HOSTS"] = "*"
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

    # ── RMSE du modèle en Production ─────────────────────────────────────
    production_rmse = None
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    for v in versions:
        if v.current_stage == "Production":
            run = client.get_run(v.run_id)
            production_rmse = run.data.metrics.get("RMSE")
            print(f"Modèle Production : version {v.version} — RMSE = {production_rmse:.4f}")
            break

    # ── RMSE du dernier modèle en Staging ────────────────────────────────
    staging_rmse = None
    staging_version = None
    for v in versions:
        if v.current_stage == "Staging":
            run = client.get_run(v.run_id)
            staging_rmse = run.data.metrics.get("RMSE")
            staging_version = v.version
            print(f"Modèle Staging   : version {v.version} — RMSE = {staging_rmse:.4f}")
            break

    if staging_rmse is None:
        print("Aucun modèle en Staging — pas de déploiement possible.")
        return "skip_promotion"

    if production_rmse is None:
        print("Aucun modèle en Production — promotion directe.")
        return "promote_to_production"

    # ── Comparaison ───────────────────────────────────────────────────────
    improvement = (production_rmse - staging_rmse) / production_rmse
    print(f"Amélioration RMSE : {improvement * 100:.2f}% (seuil : {IMPROVEMENT_THRESHOLD * 100}%)")

    if improvement >= IMPROVEMENT_THRESHOLD:
        print(f"✓ Nouveau modèle meilleur — promotion en Production")
        return "promote_to_production"
    else:
        print(f"✗ Amélioration insuffisante — modèle actuel conservé")
        return "skip_promotion"


def promote_to_production(**context):
    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

    # Trouve la version en Staging
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    staging_version = None
    for v in versions:
        if v.current_stage == "Staging":
            staging_version = v.version
            break

    if staging_version is None:
        raise ValueError("Aucune version en Staging à promouvoir")

    # Archive l'ancienne Production
    for v in versions:
        if v.current_stage == "Production":
            client.transition_model_version_stage(
                name=MODEL_NAME,
                version=v.version,
                stage="Archived",
            )
            print(f"Version {v.version} archivée")

    # Promeut Staging → Production
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=staging_version,
        stage="Production",
    )
    print(f"✓ Version {staging_version} promue en Production")


def skip_promotion(**context):
    print("Déploiement ignoré — le modèle en Production reste en place.")


with DAG(
    dag_id="dag_deploy",
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
) as dag:

    t1 = BranchPythonOperator(
        task_id="evaluate_metrics",
        python_callable=evaluate_metrics,
    )

    t2 = PythonOperator(
        task_id="promote_to_production",
        python_callable=promote_to_production,
    )

    t3 = PythonOperator(
        task_id="skip_promotion",
        python_callable=skip_promotion,
    )

    t1 >> [t2, t3]
