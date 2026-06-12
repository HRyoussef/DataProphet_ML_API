import os
import subprocess
from datetime import datetime

from airflow import DAG
from airflow.operators.python import PythonOperator


def run_retrain(**context):
    import os
    import subprocess

    env = os.environ.copy()
    env["MLFLOW_TRACKING_URI"] = "http://mlflow:5000"
    env["MLFLOW_HOST_HEADER_VALIDATION"] = "false"
    env["MLFLOW_SERVER_ALLOWED_HOSTS"] = "*"   # ← ajout
    result = subprocess.run(
        ["python", "/opt/airflow/project/retrain.py"],
        capture_output=True,
        text=True,
        env=env,
        cwd="/opt/airflow/project",
    )

    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError(f"retrain.py a échoué :\n{result.stderr}")

    print("✓ retrain.py terminé avec succès")


with DAG(
    dag_id="dag_retrain",
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
) as dag:

    t1 = PythonOperator(
        task_id="retrain_model",
        python_callable=run_retrain,
    )
