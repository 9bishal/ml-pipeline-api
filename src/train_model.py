"""
train_model.py

Purpose:
    Train the churn prediction model and save it.
    In a real MLOps setup, this script is run by CI/CD whenever the
    training data changes — the pipeline automatically retrains and
    redeploys the model.

Run:
    python src/train_model.py
"""

import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH   = os.path.join(PROJECT_ROOT, "models", "model.pkl")

# Numeric features the model uses
NUMERIC_FEATURES = ["tenure", "monthly_charges", "total_charges", "num_services"]


def generate_training_data(n: int = 2000) -> pd.DataFrame:
    """
    Generate a synthetic tabular dataset for churn prediction.
    In a real project, this would read from a database or data warehouse.
    """
    import numpy as np
    rng = np.random.default_rng(42)

    tenure           = rng.integers(1, 72, n)
    monthly_charges  = rng.uniform(20, 120, n).round(2)
    total_charges    = (tenure * monthly_charges * rng.uniform(0.8, 1.2, n)).round(2)
    num_services     = rng.integers(1, 8, n)
    contract_type    = rng.choice(["month-to-month", "one_year", "two_year"],
                                   n, p=[0.55, 0.25, 0.20])

    # Churn probability: high for short tenure + month-to-month + high charges
    churn_score = (
        (1 / (tenure + 1)) * 40
        + (monthly_charges / 120) * 30
        + (contract_type == "month-to-month").astype(float) * 20
        + rng.normal(0, 5, n)
    )
    churn = (churn_score > 30).astype(int)

    return pd.DataFrame({
        "tenure": tenure,
        "monthly_charges": monthly_charges,
        "total_charges": total_charges,
        "num_services": num_services,
        "contract_type": contract_type,
        "churn": churn,
    })


def build_and_train():
    df = generate_training_data()
    print(f"Dataset: {df.shape}, churn rate: {df.churn.mean():.1%}")

    # One-hot encode contract_type
    df = pd.get_dummies(df, columns=["contract_type"], drop_first=True)

    feature_cols = NUMERIC_FEATURES + [c for c in df.columns
                                         if c.startswith("contract_type_")]
    X = df[feature_cols]
    y = df["churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model",  LogisticRegression(class_weight="balanced",
                                       max_iter=500, random_state=42)),
    ])
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=["Stay", "Churn"]))

    # Save with feature column order — critical for serving consistency
    artifact = {"pipeline": pipeline, "feature_cols": feature_cols}
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(artifact, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    return artifact


if __name__ == "__main__":
    build_and_train()
