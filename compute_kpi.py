import json
import requests
from pathlib import Path

API_URL = "http://localhost:8000/api/predict"
HELP_DATA_DIR = Path("help_data")

# Tolérance : on accepte ±5 ans comme "correct"
TOLERANCE_YEARS = 5


def compute_kpi():
    fichiers = list(HELP_DATA_DIR.glob("*.json"))

    if not fichiers:
        print("Aucun fichier dans help_data/ — impossible de calculer le KPI.")
        return

    total = 0
    corrects = 0
    erreurs = []

    for fichier in fichiers:
        data = json.loads(fichier.read_text())

        annee_reelle = data.pop("annee_plantation_reelle")

        response = requests.post(API_URL, json=data)
        if response.status_code != 200:
            print(f"Erreur API sur {fichier.name} : {response.status_code}")
            continue

        annee_predite = response.json()["annee_plantation_predite"]
        ecart = abs(annee_predite - annee_reelle)

        total += 1
        if ecart <= TOLERANCE_YEARS:
            corrects += 1
        else:
            erreurs.append({
                "fichier": fichier.name,
                "predit": annee_predite,
                "reel": annee_reelle,
                "ecart": ecart,
            })

    precision = (corrects / total) * 100 if total > 0 else 0

    print(f"\n── KPI Niveau 3 — Précision sur feedbacks ──────────────")
    print(f"  Fichiers analysés : {total}")
    print(f"  Prédictions correctes (±{TOLERANCE_YEARS} ans) : {corrects}")
    print(f"  Précision : {precision:.1f}%")

    if erreurs:
        print(f"\n  Cas mal prédits ({len(erreurs)}) :")
        for e in erreurs:
            print(f"    {e['fichier']} — prédit {e['predit']}, réel {e['reel']} (écart {e['ecart']} ans)")


if __name__ == "__main__":
    compute_kpi()
