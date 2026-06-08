from contextlib import asynccontextmanager
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException

from schemas import CustomerFeatures, PredictionResponse

# ── Registre global du modèle ──────────────────────────────────────────────────
model = None

FEATURE_COLUMNS = [
    "genre_bota",
    "espece",
    "stadededeveloppement",
    "hauteurarbre",
    "typenature",
    "latitude",
    "longitude",
]


# ── Lifespan : chargement au démarrage, libération à l'arrêt ──────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    try:
        model = joblib.load("ml1_rf_model.pkl")
        print("✅ Modèle chargé avec succès.")
    except FileNotFoundError:
        print("❌ Fichier ml1_rf_model.pkl introuvable.")
        raise
    yield
    # -- arrêt du serveur --
    model = None
    print("🛑 Modèle libéré.")


# ── Application ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="DataProphet ML API",
    description="Prédiction de l'année de plantation d'arbres urbains.",
    version="0.2.0",
    lifespan=lifespan,
)


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/api/predict", response_model=PredictionResponse)
def predict(features: CustomerFeatures):
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé.")

    # Pydantic → DataFrame (le pipeline attend un DataFrame, pas un numpy array)
    df = pd.DataFrame([features.model_dump()], columns=FEATURE_COLUMNS)

    prediction = model.predict(df)[0]

    return PredictionResponse(annee_plantation_predite=round(float(prediction), 1))
