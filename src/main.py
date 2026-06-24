"""
main.py

FastAPI application exposing the churn prediction model as a REST API.

Endpoints:
    GET  /health   - liveness check + model status
    POST /predict  - predict churn for a single customer

Run locally:
    uvicorn src.main:app --reload --port 8000

Then visit http://localhost:8000/docs for the interactive Swagger UI.
"""

import os
import sys
from contextlib import asynccontextmanager

# pyrefly: ignore [missing-import]
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException

# Allow importing from src/ when running from project root (e.g. uvicorn)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from schemas import PredictRequest, PredictResponse, HealthResponse

MODEL_PATH   = os.path.join(PROJECT_ROOT, "models", "model.pkl")

# Global model artifact — loaded once at startup, shared across all requests
_artifact = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load the model when the app starts, release on shutdown.
    Using lifespan (instead of @app.on_event) is the modern FastAPI pattern.
    Loading once at startup is critical for performance — joblib.load() is
    slow; we never want it inside the request handler.
    """
    global _artifact
    if not os.path.exists(MODEL_PATH):
        print(f"Model file not found at {MODEL_PATH}. Run 'python src/train_model.py' to train and save it.")
        _artifact = None
        yield
        return

    try:
        _artifact = joblib.load(MODEL_PATH)
        print(f"Model loaded from {MODEL_PATH}")
    except Exception as e:
        print(f"Failed to load model: {e}")
        _artifact = None
        yield
        return
    
    # Backward compatibility patch for LogisticRegression unpickled in different scikit-learn versions
    if _artifact and "pipeline" in _artifact:
        pipeline = _artifact["pipeline"]
        if "model" in pipeline.named_steps:
            final_estimator = pipeline.named_steps["model"]
            if type(final_estimator).__name__ == "LogisticRegression" and not hasattr(final_estimator, "multi_class"):
                final_estimator.multi_class = "auto"
                
    yield
    _artifact = None


app = FastAPI(
    title="Churn Prediction API",
    description="Predicts customer churn probability for a subscription business.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health():
    """Liveness check — used by load balancers and Docker HEALTHCHECK."""
    return HealthResponse(status="ok", model_loaded=_artifact is not None)


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
def predict(request: PredictRequest):
    """
    Predict churn for a single customer.

    FastAPI automatically:
    - Validates the incoming JSON against PredictRequest
    - Returns a 422 error with clear field-level messages if validation fails
    - Serializes the return value using PredictResponse
    """
    if _artifact is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    pipeline     = _artifact["pipeline"]
    feature_cols = _artifact["feature_cols"]

    # Build a one-row DataFrame in exactly the shape the model expects
    row = {
        "tenure":           request.tenure,
        "monthly_charges":  request.monthly_charges,
        "total_charges":    request.total_charges,
        "num_services":     request.num_services,
        "contract_type_one_year": 1 if request.contract_type == "one_year"  else 0,
        "contract_type_two_year": 1 if request.contract_type == "two_year"  else 0,
    }

    X = pd.DataFrame([row]).reindex(columns=feature_cols, fill_value=0)

    prob       = float(pipeline.predict_proba(X)[0][1])
    prediction = int(prob >= 0.5)
    label      = "Churn" if prediction == 1 else "Stay"
    risk_tier  = "High" if prob >= 0.6 else ("Medium" if prob >= 0.3 else "Low")

    return PredictResponse(
        churn_prediction=prediction,
        churn_label=label,
        churn_probability=round(prob, 4),
        risk_tier=risk_tier,
    )
