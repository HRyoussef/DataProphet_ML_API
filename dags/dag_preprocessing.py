import json
import pandas as pd
from pathlib import Path
from datetime import datetime

from airflow import DAG
from airflow.operators.python import PythonOperator

# Chemins relatifs au projet monté dans /opt/airflow/project
HELP_DATA_DIR = Path("/opt/airflow/project/help_data")
OUTPUT_DIR    = Path("/opt/airflow/project/data")

EXPECTED_FEATURES = [
    "genre_bota",
    "espece",
    "stadededeveloppement",
    "hauteurarbre",
    "typenature",
    "latitude",
    "longitude",
    "annee_plantation_reelle",
]


def load_and_validate(**context):
    fichiers = list(HELP_DATA_DIR.glob("*.json"))

    if not fichiers:
        print("help_data/ est vide — rien à traiter.")
        return []

    records_valides = []
    rejets = 0

    for fichier in fichiers:
        try:
            data = json.loads(fichier.read_text())
            if all(k in data for k in EXPECTED_FEATURES):
                records_valides.append(data)
            else:
                manquantes = [k for k in EXPECTED_FEATURES if k not in data]
                print(f"Rejeté {fichier.name} — champs manquants : {manquantes}")
                rejets += 1
        except json.JSONDecodeError:
            print(f"Rejeté {fichier.name} — JSON invalide")
            rejets += 1

    print(f"Résultat : {len(records_valides)} valides, {rejets} rejetés")
    return records_valides 


def prepare_dataset(**context):
    records = context["ti"].xcom_pull(
        task_ids="load_and_validate",
        key="return_value"
    )

    if not records:
        print("Aucune donnée validée — dataset non généré.")
        return

    OUTPUT_DIR.mkdir(exist_ok=True)

    df = pd.DataFrame(records)

    # Renomme la colonne label pour correspondre à ce qu'attend retrain.py
    df = df.rename(columns={"annee_plantation_reelle": "anneedeplantation"})

    output_path = OUTPUT_DIR / "retrain_dataset.csv"
    df.to_csv(output_path, index=False)
    print(f"Dataset sauvegardé : {output_path} ({len(df)} lignes)")


with DAG(
    dag_id="dag_preprocessing",
    start_date=datetime(2024, 1, 1),
    schedule=None,        # déclenché manuellement ou par dag_mlops_weekly
    catchup=False,
) as dag:

    t1 = PythonOperator(
        task_id="load_and_validate",
        python_callable=load_and_validate,
    )

    t2 = PythonOperator(
        task_id="prepare_dataset",
        python_callable=prepare_dataset,
    )

    t1 >> t2
