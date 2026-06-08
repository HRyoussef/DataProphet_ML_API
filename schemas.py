from pydantic import BaseModel, Field
from typing import Literal


class CustomerFeatures(BaseModel):
    age: int = Field(..., ge=18, le=100, description="Âge du client")
    tenure: int = Field(..., ge=0, description="Nombre de mois comme client")
    monthly_charges: float = Field(..., gt=0, description="Charges mensuelles en €")
    total_charges: float = Field(..., ge=0, description="Total facturé depuis le début")
    num_products: int = Field(..., ge=1, le=10, description="Nombre de produits souscrits")
    has_support_contract: int = Field(..., ge=0, le=1, description="Contrat support : 1=oui, 0=non")
    num_complaints: int = Field(..., ge=0, description="Nombre de réclamations")

    model_config = {
        "json_schema_extra": {
            "example": {
                "age": 35,
                "tenure": 24,
                "monthly_charges": 65.90,
                "total_charges": 1581.60,
                "num_products": 2,
                "has_support_contract": 1,
                "num_complaints": 0,
            }
        }
    }


class PredictionResponse(BaseModel):
    churn: int = Field(..., description="Prédiction brute : 1=churn, 0=pas de churn")
    label: Literal["Churn", "No Churn"] = Field(..., description="Label lisible")
    probability: float = Field(..., ge=0.0, le=1.0, description="Probabilité de churn")
