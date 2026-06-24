# ML Pipeline API — FastAPI + Docker + CI

A production-grade ML model served via a **FastAPI REST API**, containerized
with **Docker**, and tested with **pytest** running in a **GitHub Actions** CI
pipeline on every push.

## Problem Statement

A trained churn prediction model sitting in a Jupyter notebook has zero
production value. This project shows how to take that model and expose it
as a REST endpoint that any application (mobile app, CRM, dashboard) can
call — with input validation, automatic API docs, containerization, and a
test suite that runs on every code change.

## Architecture

```
src/
├── train_model.py   # trains model, saves models/model.pkl
├── schemas.py        # Pydantic request/response models (validation)
└── main.py           # FastAPI app: /health + /predict endpoints
tests/
└── test_api.py       # 6 pytest tests (runs in CI on every push)
models/
└── model.pkl         # trained sklearn Pipeline artifact
Dockerfile            # multi-stage build (builder + runtime)
.github/workflows/
└── ci.yml            # GitHub Actions: train → test → lint on push
```

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Liveness check + model status |
| POST | `/predict` | Predict churn for a customer |
| GET | `/docs` | Interactive Swagger UI (auto-generated) |

### Sample Request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "tenure": 3,
    "monthly_charges": 95.0,
    "total_charges": 285.0,
    "num_services": 2,
    "contract_type": "month-to-month"
  }'
```

### Sample Response

```json
{
  "churn_prediction": 1,
  "churn_label": "Churn",
  "churn_probability": 0.8241,
  "risk_tier": "High"
}
```

## How to Run

### Locally

```bash
pip install -r requirements.txt
python src/train_model.py         # train + save model
pytest tests/ -v                   # run test suite
uvicorn src.main:app --reload --host  [IP_ADDRESS] --port 8000     # start API at http://localhost:8000
```

### With Docker

```bash
docker build -t churn-api .
docker run -p 8000:8000 churn-api
# API available at http://localhost:8000/docs
```

## Test Results

```
tests/test_api.py::test_health_endpoint          PASSED
tests/test_api.py::test_predict_valid_input      PASSED
tests/test_api.py::test_predict_low_risk_customer PASSED
tests/test_api.py::test_predict_high_risk_customer PASSED
tests/test_api.py::test_predict_invalid_tenure   PASSED
tests/test_api.py::test_predict_missing_field    PASSED
6 passed in 0.92s
```

## Key Design Decisions

- **Pydantic validation**: FastAPI automatically returns 422 errors with
  field-level messages on bad input — no manual validation code needed
- **Model loaded once at startup** via `lifespan`: `joblib.load()` is slow;
  loading per-request would kill performance
- **Multi-stage Dockerfile**: builder installs dependencies, runtime stage
  copies only what's needed — keeps image lean and secure
- **Non-root Docker user**: basic security practice, often checked in
  production deployments
- **CI retrains the model**: ensures the model artifact is always fresh
  and tests always run against a working model

## Tech Stack

Python · FastAPI · Pydantic v2 · Scikit-learn · Docker · pytest · GitHub Actions
