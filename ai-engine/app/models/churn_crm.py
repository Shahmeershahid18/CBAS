"""
CRM-native Churn — inference wrapper. Loads the model trained on CBAS's own data
(training/churn_crm.py) and scores one contact's lapse risk from engagement
features. Distinct from app/models/churn.py, which serves the Telco benchmark.
"""
from __future__ import annotations

import threading

import joblib
import numpy as np
import pandas as pd

from app.config import CHURN_CRM_ARTIFACT

_lock = threading.Lock()
_bundle: dict | None = None

FEATURES = [
    "tenure_days", "num_orders", "total_spend", "num_activities",
    "num_negative_notes", "avg_sentiment", "num_categories", "has_open_deal",
]


class ModelNotTrainedError(RuntimeError):
    pass


def _load() -> dict:
    global _bundle
    if _bundle is None:
        with _lock:
            if _bundle is None:
                if not CHURN_CRM_ARTIFACT.exists():
                    raise ModelNotTrainedError(
                        f"No trained model at {CHURN_CRM_ARTIFACT}. "
                        "Export + train: npx tsx scripts/export-ai-training.ts && python -m training.churn_crm"
                    )
                _bundle = joblib.load(CHURN_CRM_ARTIFACT)
    return _bundle


def is_ready() -> bool:
    return CHURN_CRM_ARTIFACT.exists()


def model_info() -> dict:
    b = _load()
    return {
        "model_name": b["model_name"], "data_source": b["data_source"],
        "metrics": b["metrics"], "top_features": b["top_features"], "trained_rows": b["trained_rows"],
    }


def _band(prob: float) -> str:
    return "High" if prob >= 0.60 else "Medium" if prob >= 0.30 else "Low"


def predict_churn(features: dict) -> dict:
    bundle = _load()
    row = {f: features.get(f, np.nan) for f in FEATURES}
    X = pd.DataFrame([row], columns=FEATURES)
    prob = float(bundle["pipeline"].predict_proba(X)[0, 1])
    score = int(round(prob * 100))
    band = _band(prob)
    top = [t["feature"] for t in bundle["top_features"][:3]]
    reason = (f"{band} churn risk: {score}/100. Model ({bundle['model_name']}, trained on CRM data) "
              f"weighs {', '.join(top)} most heavily.")
    return {
        "churn_probability": round(prob, 4), "risk_score": score, "risk_band": band,
        "model": bundle["model_name"], "reason": reason, "key_factors": bundle["top_features"][:5],
    }
