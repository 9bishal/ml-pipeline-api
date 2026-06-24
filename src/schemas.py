"""
schemas.py

Pydantic models for request validation and response formatting.
FastAPI uses these to automatically validate incoming JSON, return
clean error messages on bad input, and generate the /docs OpenAPI page.
"""

from typing import Literal

from pydantic import BaseModel, Field, ConfigDict


class PredictRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "tenure": 3,
                "monthly_charges": 95.0,
                "total_charges": 285.0,
                "num_services": 2,
                "contract_type": "month-to-month",
            }
        }
    )

    tenure: int = Field(..., ge=0, le=72, description="Months with company (0-72)")
    monthly_charges: float = Field(..., ge=0, le=200, description="Monthly bill amount")
    total_charges: float = Field(..., ge=0, description="Total billed to date")
    num_services: int = Field(..., ge=1, le=8, description="Number of subscribed services")
    contract_type: Literal["month-to-month", "one_year", "two_year"] = Field(..., description="Contract type")


class PredictResponse(BaseModel):
    churn_prediction: int          # 0 = stay, 1 = churn
    churn_label: str               # "Stay" or "Churn"
    churn_probability: float       # probability of churning
    risk_tier: str                 # Low / Medium / High


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
