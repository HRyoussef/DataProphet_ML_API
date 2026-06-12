from datetime import datetime
from airflow import DAG
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

with DAG(
    dag_id="dag_mlops_weekly",
    start_date=datetime(2024, 1, 1),
    schedule="0 6 * * 1",   # chaque lundi à 6h00
    catchup=False,
) as dag:

    t1 = TriggerDagRunOperator(
        task_id="trigger_preprocessing",
        trigger_dag_id="dag_preprocessing",
        wait_for_completion=True,
    )

    t2 = TriggerDagRunOperator(
        task_id="trigger_retrain",
        trigger_dag_id="dag_retrain",
        wait_for_completion=True,
    )

    t3 = TriggerDagRunOperator(
        task_id="trigger_deploy",
        trigger_dag_id="dag_deploy",
        wait_for_completion=True,
    )

    t1 >> t2 >> t3
