"""
Synthetic customer generator for churn — mirrors the shape of the Kaggle Telco
Customer Churn dataset so the pipeline runs end-to-end without the download.
Clearly synthetic: train on the real Telco CSV for the FYP defence.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.models.churn_features import (
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET,
)

CONTRACTS = ["Month-to-month", "One year", "Two year"]
PAYMENT_METHODS = ["Electronic check", "Mailed check",
                   "Bank transfer (automatic)", "Credit card (automatic)"]
INTERNET = ["DSL", "Fiber optic", "No"]
YES_NO = ["Yes", "No"]


def generate_customers(n: int = 7043, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    tenure = rng.integers(0, 73, n).astype(float)                # months
    monthly_charges = rng.uniform(18.0, 120.0, n)
    # Total charges roughly tenure * monthly with noise.
    total_charges = np.maximum(0.0, tenure * monthly_charges * rng.uniform(0.8, 1.1, n))

    contract = rng.choice(CONTRACTS, n, p=[0.55, 0.21, 0.24])
    payment_method = rng.choice(PAYMENT_METHODS, n)
    internet = rng.choice(INTERNET, n, p=[0.34, 0.44, 0.22])
    paperless = rng.choice(YES_NO, n, p=[0.59, 0.41])
    tech_support = rng.choice(YES_NO, n)
    online_security = rng.choice(YES_NO, n)
    senior = rng.choice(["0", "1"], n, p=[0.84, 0.16])

    df = pd.DataFrame({
        "tenure": tenure,
        "monthly_charges": monthly_charges,
        "total_charges": total_charges,
        "contract": contract,
        "payment_method": payment_method,
        "internet_service": internet,
        "paperless_billing": paperless,
        "tech_support": tech_support,
        "online_security": online_security,
        "senior_citizen": senior,
    })

    # Latent churn propensity: short tenure, month-to-month, fiber, e-check,
    # no support all push churn up (matches the real Telco patterns).
    z = (
        -0.5
        - 0.045 * df["tenure"]
        + 0.010 * df["monthly_charges"]
        + np.where(df["contract"] == "Month-to-month", 1.4, 0.0)
        + np.where(df["contract"] == "Two year", -1.2, 0.0)
        + np.where(df["internet_service"] == "Fiber optic", 0.7, 0.0)
        + np.where(df["payment_method"] == "Electronic check", 0.6, 0.0)
        + np.where(df["tech_support"] == "No", 0.4, 0.0)
        + np.where(df["senior_citizen"] == "1", 0.3, 0.0)
    )
    prob = 1.0 / (1.0 + np.exp(-z))
    df[TARGET] = (rng.random(n) < prob).astype(int)

    # A few blank total_charges like the real set (new customers, tenure 0).
    df.loc[df["tenure"] == 0, "total_charges"] = np.nan

    return df[NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET]]
