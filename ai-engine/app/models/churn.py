"""
Churn Prediction — inference wrapper.

Loads the joblib bundle from `training.churn` and turns one customer's features
into a churn probability + risk band (Low / Medium / High). Tolerant of missing
fields — anything omitted is imputed by the pipeline.
"""
from __future__ import annotations

import threading

import joblib
import numpy as np
import pandas as pd

from app.config import CHURN_ARTIFACT
from app.models.churn_features import ALL_FEATURES

_lock = threading.Lock()
_bundle: dict | None = None


class ModelNotTrainedError(RuntimeError):
    """Raised when no trained artifact exists yet."""


def _load() -> dict:
    global _bundle
    if _bundle is None:
        with _lock:
            if _bundle is None:
                if not CHURN_ARTIFACT.exists():
                    raise ModelNotTrainedError(
                        f"No trained model at {CHURN_ARTIFACT}. "
                        "Run: python -m training.churn"
                    )
                _bundle = joblib.load(CHURN_ARTIFACT)
    return _bundle


def is_ready() -> bool:
    return CHURN_ARTIFACT.exists()


def model_info() -> dict:
    b = _load()
    return {
        "model_name": b["model_name"],
        "data_source": b["data_source"],
        "metrics": b["metrics"],
        "top_features": b["top_features"],
        "trained_rows": b["trained_rows"],
    }


def _band(prob: float) -> str:
    if prob >= 0.60:
        return "High"
    if prob >= 0.30:
        return "Medium"
    return "Low"


def predict_churn(features: dict) -> dict:
    """
    features: partial/complete dict keyed by app.models.churn_features.ALL_FEATURES.
    Returns: {churn_probability, risk_score, risk_band, model, reason, key_factors}
    """
    bundle = _load()
    pipeline = bundle["pipeline"]

    row = {f: features.get(f, np.nan) for f in ALL_FEATURES}
    X = pd.DataFrame([row], columns=ALL_FEATURES)

    prob = float(pipeline.predict_proba(X)[0, 1])
    risk_score = int(round(prob * 100))
    band = _band(prob)

    provided = [f for f in ALL_FEATURES if f in features and features[f] not in (None, "")]
    top = [t["feature"] for t in bundle["top_features"][:3]]
    reason = (
        f"{band} churn risk: {risk_score}/100 probability of leaving. "
        f"Model ({bundle['model_name']}) weighs {', '.join(top)} most heavily. "
        f"Scored from {len(provided)} provided field(s)."
    )

    return {
        "churn_probability": round(prob, 4),
        "risk_score": risk_score,
        "risk_band": band,
        "model": bundle["model_name"],
        "reason": reason,
        "key_factors": bundle["top_features"][:5],
    }
