"""
tests/test_api.py

Pytest tests for the churn prediction API.

Why write tests for a portfolio project?
- Most freshers skip tests entirely. Having a test file signals production
  mindset and makes the CI pipeline meaningful (tests run on every push).
- The tests cover the two most important behaviours: successful prediction
  and validation rejection on bad input.

Run:
    pytest tests/ -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
from fastapi.testclient import TestClient
from main import app

VALID_PAYLOAD = {
    "tenure": 3,
    "monthly_charges": 95.0,
    "total_charges": 285.0,
    "num_services": 2,
    "contract_type": "month-to-month",
}


@pytest.fixture(scope="module")
def client():
    """
    Use TestClient as a context manager so the FastAPI lifespan
    (model loading) is triggered before tests run.
    """
    with TestClient(app) as c:
        yield c


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True


def test_predict_valid_input(client):
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200
    data = response.json()
    assert data["churn_prediction"] in [0, 1]
    assert 0.0 <= data["churn_probability"] <= 1.0
    assert data["risk_tier"] in ["Low", "Medium", "High"]
    assert data["churn_label"] in ["Stay", "Churn"]


def test_predict_low_risk_customer(client):
    response = client.post("/predict", json={
        "tenure": 60, "monthly_charges": 45.0,
        "total_charges": 2700.0, "num_services": 5,
        "contract_type": "two_year",
    })
    assert response.status_code == 200
    data = response.json()
    assert data["churn_prediction"] == 0
    assert data["risk_tier"] in ["Low", "Medium"]


def test_predict_high_risk_customer(client):
    response = client.post("/predict", json={
        "tenure": 1, "monthly_charges": 115.0,
        "total_charges": 115.0, "num_services": 1,
        "contract_type": "month-to-month",
    })
    assert response.status_code == 200
    assert response.json()["churn_prediction"] == 1


def test_predict_invalid_tenure(client):
    response = client.post("/predict", json={**VALID_PAYLOAD, "tenure": -5})
    assert response.status_code == 422


def test_predict_missing_field(client):
    incomplete = {k: v for k, v in VALID_PAYLOAD.items() if k != "contract_type"}
    response = client.post("/predict", json=incomplete)
    assert response.status_code == 422
