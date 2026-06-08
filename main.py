from fastapi import FastAPI
from schemas import CustomerFeatures, PredictionResponse

app = FastAPI(
    title="DataProphet ML API",
    description="Service ML exposant des modèles de prédiction via HTTP.",
    version="0.1.0",
)


@app.get("/health")
def health():
    return {"status": "ok", "message": "Le service est actif."}


@app.post("/api/predict", response_model=PredictionResponse)
def predict(customer: CustomerFeatures):
    # 🔧 Prédiction factice — le vrai modèle sera branché à l'étape suivante
    churn = 1
    probability = 0.87

    return PredictionResponse(
        churn=churn,
        label="Churn" if churn == 1 else "No Churn",
        probability=probability,
    )
