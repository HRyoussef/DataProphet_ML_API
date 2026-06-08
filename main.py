from fastapi import FastAPI

app = FastAPI(
    title="DataProphet ML API",
    description="Service ML exposant des modèles de prédiction via HTTP.",
    version="0.1.0",
)


@app.get("/health")
def health():
    return {"status": "ok", "message": "Le service est actif."}
