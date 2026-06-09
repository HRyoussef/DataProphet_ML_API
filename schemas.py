from pydantic import BaseModel, Field
from typing import Optional, Literal


class CustomerFeatures(BaseModel):
    genre_bota: str = Field(..., description="Genre botanique de l'arbre", example="Prunus")
    espece: str = Field(..., description="Espèce de l'arbre", example="serrulata")
    stadededeveloppement: str = Field(..., description="Stade de développement", example="Arbre jeune")
    hauteurarbre: Optional[str] = Field(None, description="Hauteur de l'arbre (ex: 'de 10 m à 20 m')")
#hauteurarbre: Optional[float] = Field(None, description="Hauteur de l'arbre en mètres")
    typenature: Optional[str] = Field(None, description="Type de nature")
    latitude: float = Field(..., ge=-90, le=90, description="Latitude GPS", example=45.167098)
    longitude: float = Field(..., ge=-180, le=180, description="Longitude GPS", example=5.740132)

    model_config = {
        "json_schema_extra": {
            "example": {
                "genre_bota": "Prunus",
                "espece": "serrulata",
                "stadededeveloppement": "Arbre jeune",
                "hauteurarbre": None,
                "typenature": None,
                "latitude": 45.167098,
                "longitude": 5.740132,
            }
        }
    }


class PredictionResponse(BaseModel):
    annee_plantation_predite: float = Field(..., description="Année de plantation prédite par le modèle")
