"""CRM-native Lead Scoring — inference wrapper (model trained on CBAS's leads)."""
from __future__ import annotations

import threading

import joblib
import numpy as np
import pandas as pd

from app.config import LEAD_CRM_ARTIFACT

_lock = threading.Lock()
_bundle: dict | None = None

FEATURES = ["quotation", "num_activities", "has_email", "has_phone", "source", "service"]


class ModelNotTrainedError(RuntimeError):
    pass


def _load() -> dict:
    global _bundle
    if _bundle is None:
        with _lock:
            if _bundle is None:
                if not LEAD_CRM_ARTIFACT.exists():
                    raise ModelNotTrainedError(
                        f"No trained model at {LEAD_CRM_ARTIFACT}. "
                        "Export + train: npx tsx scripts/export-ai-training.ts && python -m training.lead_crm")
                _bundle = joblib.load(LEAD_CRM_ARTIFACT)
    return _bundle


def is_ready() -> bool:
    return LEAD_CRM_ARTIFACT.exists()


def model_info() -> dict:
    b = _load()
    return {"model_name": b["model_name"], "data_source": b["data_source"],
            "metrics": b["metrics"], "top_features": b["top_features"], "trained_rows": b["trained_rows"]}


def _band(score: int) -> str:
    return "Hot" if score >= 70 else "Warm" if score >= 40 else "Cold"


def score_lead(features: dict) -> dict:
    bundle = _load()
    row = {f: features.get(f, np.nan) for f in FEATURES}
    X = pd.DataFrame([row], columns=FEATURES)
    prob = float(bundle["pipeline"].predict_proba(X)[0, 1])
    score = int(round(prob * 100))
    band = _band(score)
    top = [t["feature"] for t in bundle["top_features"][:3]]
    reason = (f"{band} lead: {score}/100 conversion probability. "
              f"Model ({bundle['model_name']}, trained on CRM data) weighs {', '.join(top)} most heavily.")
    return {"score": score, "probability": round(prob, 4), "band": band,
            "model": bundle["model_name"], "reason": reason, "key_factors": bundle["top_features"][:5]}
