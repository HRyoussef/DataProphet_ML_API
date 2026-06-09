# main.py
import mlflow.sklearn
import mlflow
import pandas as pd
from fastapi import FastAPI
from contextlib import asynccontextmanager
#from schemas import TreeInput, PredictionOutput
from schemas import CustomerFeatures, PredictionResponse
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

@app.get("/health")
def health():
    return {"status": "ok", "model": "DataProphet/Production"}

@app.post("/api/predict", response_model=PredictionResponse)
def predict(data: CustomerFeatures):
    df = pd.DataFrame([data.model_dump()])
    prediction = model.predict(df)
    return PredictionResponse(annee_plantation_predite=round(float(prediction[0]), 2))
