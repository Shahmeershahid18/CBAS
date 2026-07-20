"""
Lead Scoring — inference wrapper.

Loads the joblib bundle produced by `training.lead_scoring` and turns a single
lead's raw features into a 0-100 score. Tolerant of missing fields: any feature
the caller omits is passed as NaN and handled by the pipeline's imputers, so the
CRM can send whatever it happens to know about a lead.
"""
from __future__ import annotations

import threading

import joblib
import numpy as np
import pandas as pd

from app.config import LEAD_SCORING_ARTIFACT
from app.models.lead_features import ALL_FEATURES

_lock = threading.Lock()
_bundle: dict | None = None


class ModelNotTrainedError(RuntimeError):
    """Raised when no trained artifact exists yet."""


def _load() -> dict:
    global _bundle
    if _bundle is None:
        with _lock:
            if _bundle is None:
                if not LEAD_SCORING_ARTIFACT.exists():
                    raise ModelNotTrainedError(
                        f"No trained model at {LEAD_SCORING_ARTIFACT}. "
                        "Run: python -m training.lead_scoring"
                    )
                _bundle = joblib.load(LEAD_SCORING_ARTIFACT)
    return _bundle


def is_ready() -> bool:
    return LEAD_SCORING_ARTIFACT.exists()


def model_info() -> dict:
    b = _load()
    return {
        "model_name": b["model_name"],
        "data_source": b["data_source"],
        "metrics": b["metrics"],
        "top_features": b["top_features"],
        "trained_rows": b["trained_rows"],
    }


def _band(score: int) -> str:
    if score >= 70:
        return "Hot"
    if score >= 40:
        return "Warm"
    return "Cold"


def score_lead(features: dict) -> dict:
    """
    features: partial/complete dict keyed by the internal feature names
              (see app.models.lead_features.ALL_FEATURES).
    Returns: {score, probability, band, model, reason, key_factors}
    """
    bundle = _load()
    pipeline = bundle["pipeline"]

    # Build a single-row frame with every expected column; missing -> NaN.
    row = {f: features.get(f, np.nan) for f in ALL_FEATURES}
    X = pd.DataFrame([row], columns=ALL_FEATURES)

    prob = float(pipeline.predict_proba(X)[0, 1])
    score = int(round(prob * 100))
    band = _band(score)

    provided = [f for f in ALL_FEATURES if f in features and features[f] not in (None, "")]
    top = [t["feature"] for t in bundle["top_features"][:3]]
    reason = (
        f"{band} lead: {score}/100 conversion probability. "
        f"Model ({bundle['model_name']}) weighs {', '.join(top)} most heavily. "
        f"Scored from {len(provided)} provided field(s)."
    )

    return {
        "score": score,
        "probability": round(prob, 4),
        "band": band,
        "model": bundle["model_name"],
        "reason": reason,
        "key_factors": bundle["top_features"][:5],
    }
