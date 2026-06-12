# main.py

import uuid
import json
from pathlib import Path
import time
from fastapi.responses import Response                               # ← nouveau
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST  # ← nouveau
from prometheus_fastapi_instrumentator import Instrumentator
from metrics import PREDICTION_COUNTER, PREDICTION_DURATION         # ← nouveau
import mlflow.sklearn
import mlflow
import pandas as pd
from fastapi import FastAPI
from contextlib import asynccontextmanager
#from schemas import TreeInput, PredictionOutput
from schemas import CustomerFeatures, PredictionResponse, HelpData


HELP_DATA_DIR = Path("help_data")
HELP_DATA_DIR.mkdir(exist_ok=True)   # crée le dossier si inexistant
# ─────────────────────────────────────────────
# Lifespan : chargement du modèle au démarrage
# ─────────────────────────────────────────────
model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model

    # Connexion au serveur MLflow
    mlflow.set_tracking_uri("http://localhost:5000")

    # Chargement depuis le Registry — stage Production
    print("⏳ Chargement du modèle depuis MLflow Registry...")
    model = mlflow.sklearn.load_model("models:/DataProphet/Production")
    #Instrumentator().instrument(app).expose(app, endpoint="/metrics")
    print("✅ Modèle chargé depuis MLflow Registry (DataProphet/Production)")

    yield  # l'API est prête

    print("🛑 Arrêt de l'API")

# ─────────────────────────────────────────────
# Application FastAPI
# ─────────────────────────────────────────────
app = FastAPI(
    title       = "DataProphet ML API",
    description = "Prédiction de l'année de plantation des arbres de Grenoble",
    version     = "2.0.0",
    lifespan    = lifespan,
)
Instrumentator().instrument(app).expose(app, endpoint="/metrics-system")
@app.get("/health")
def health():
    return {"status": "ok", "model": "DataProphet/Production"}

@app.post("/api/predict", response_model=PredictionResponse)
def predict(data: CustomerFeatures):
    df = pd.DataFrame([data.model_dump()])
    start = time.time()
    prediction = model.predict(df)
    duration = time.time() - start
    # Calcul de la décennie pour le label Counter
    # ex: 1994 → "1990s",  2003 → "2000s"
#    decade = f"{int(prediction.item()) // 10 * 10}s"                # ← nouveau
    # Arbre planté avant 2000 → "vieux", après → "jeune"
    age_category = "jeune" if data.hauteurarbre in ["de 0 m à 5 m", "de 5 m à 10 m"] else "vieux"

    PREDICTION_COUNTER.labels(age_category=age_category).inc()
    PREDICTION_DURATION.observe(duration)                           # ← nouveau
    time.sleep(0.6) 
    return PredictionResponse(annee_plantation_predite=round(float(prediction[0]), 2))
@app.get("/metrics")                                                 # ← nouveau
def metrics():                                                       # ← nouveau
    return Response(                                                 # ← nouveau
        content=generate_latest(),                                   # ← nouveau
        media_type=CONTENT_TYPE_LATEST,                             # ← nouveau
    )


@app.post("/api/helpdata")
def collect_feedback(data: HelpData):
    fichier = HELP_DATA_DIR / f"{uuid.uuid4()}.json"
    fichier.write_text(json.dumps(data.model_dump(), indent=2))
    return {"status": "saved", "fichier": fichier.name}
