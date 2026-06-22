# DataProphet ML API

Pipeline MLOps complet pour la prédiction de l'année de plantation des arbres urbains de Grenoble — du modèle scikit-learn à la production, avec versioning, monitoring et automatisation.

## Stack technique

| Catégorie | Outils |
|---|---|
| API & ML | FastAPI, Pydantic v2, scikit-learn, RandomForestRegressor |
| MLOps | MLflow (Registry + Tracking), PostgreSQL, RustFS (S3-compatible) |
| Monitoring | Prometheus, Grafana, prometheus-fastapi-instrumentator |
| Orchestration | Apache Airflow (LocalExecutor) |
| Infrastructure | Docker Compose |

## Architecture

```
FastAPI :8000  ──predict──→  MLflow Registry :5000
     │                              │
     ├──/metrics──→ Prometheus :9090 ──→ Grafana :3000
     │
     └──/api/helpdata──→ help_data/ ──→ Airflow DAGs :8080
```

## Fonctionnalités

### API (`main.py`)
- `POST /api/predict` — prédiction de l'année de plantation
- `POST /api/helpdata` — collecte de corrections terrain (feedback loop)
- `GET /metrics` — métriques custom Prometheus (Counter, Histogram)
- `GET /metrics-system` — métriques système (latence, codes HTTP)

### MLOps (MLflow)
- `train.py` — entraînement initial avec tracking MLflow
- `retrain.py` — réentraînement fusionnant les données de feedback, promotion conditionnelle (seuil RMSE 2%)
- `promote_model.py` — gestion manuelle du cycle de vie du Registry

### Monitoring (Prometheus + Grafana)
Dashboard **DataProphet — Monitoring Production**, 5 panels :
- Volume de requêtes (req/min)
- Latence moyenne
- Répartition jeunes/vieux (Pie chart)
- Latence P95 (`histogram_quantile`)
- Taux d'erreurs 5xx

2 alertes configurées : dérive du modèle (>80% de prédictions homogènes sur 5min), latence P95 dégradée (>seuil sur 2min).

### Automatisation (Airflow)
4 DAGs orchestrant le cycle MLOps hebdomadaire :

| DAG | Rôle |
|---|---|
| `dag_preprocessing` | Valide et normalise les feedbacks (`help_data/` → `data/retrain_dataset.csv`) |
| `dag_retrain` | Déclenche `retrain.py`, vérifie l'apparition d'un nouveau run MLflow |
| `dag_deploy` | `BranchPythonOperator` — compare RMSE Staging/Production, promotion conditionnelle |
| `dag_mlops_weekly` | Orchestre les 3 DAGs en séquence, planifié chaque lundi 6h |

### KPI métier (`compute_kpi.py`)
Calcule la précision réelle du modèle en comparant ses prédictions aux corrections terrain collectées via `/api/helpdata`.

## Installation

```bash
conda env create -f environment.yml
conda activate dataprophet
cp .env.example .env  # à compléter
docker-compose up -d
```

Services exposés : MLflow `:5000`, Prometheus `:9090`, Grafana `:3000`, Airflow `:8080`.

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## Entraînement initial

```bash
python train.py
python promote_model.py
```

## KPI de précision

```bash
python compute_kpi.py
```

## Structure du projet

```
.
├── main.py                 # API FastAPI
├── schemas.py               # Schémas Pydantic
├── metrics.py                # Métriques Prometheus custom
├── train.py / retrain.py     # Entraînement / réentraînement
├── promote_model.py          # Gestion Registry MLflow
├── compute_kpi.py            # KPI métier
├── dags/                     # 4 DAGs Airflow
├── docker-compose.yml
├── prometheus.yml
└── environment.yml
```
