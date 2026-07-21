"""CRM-note Sentiment — inference wrapper (model trained on business language)."""
from __future__ import annotations

import re
import threading

import joblib

from app.config import SENTIMENT_CRM_ARTIFACT

_lock = threading.Lock()
_bundle: dict | None = None
_clean_re = re.compile(r"[^a-z0-9\s]")


class ModelNotTrainedError(RuntimeError):
    pass


def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", _clean_re.sub(" ", str(s).lower())).strip()


def _load() -> dict:
    global _bundle
    if _bundle is None:
        with _lock:
            if _bundle is None:
                if not SENTIMENT_CRM_ARTIFACT.exists():
                    raise ModelNotTrainedError(
                        f"No trained model at {SENTIMENT_CRM_ARTIFACT}. Run: python -m training.sentiment_crm")
                _bundle = joblib.load(SENTIMENT_CRM_ARTIFACT)
    return _bundle


def is_ready() -> bool:
    return SENTIMENT_CRM_ARTIFACT.exists()


def model_info() -> dict:
    b = _load()
    return {"model_name": b["model_name"], "data_source": b["data_source"],
            "labels": b["labels"], "metrics": {"accuracy": b["metrics"]["accuracy"],
            "macro_f1": b["metrics"]["macro_f1"]}, "trained_rows": b["trained_rows"]}


def analyze_sentiment(text: str) -> dict:
    bundle = _load()
    pipeline = bundle["pipeline"]
    cleaned = _clean(text)
    if not cleaned:
        return {"sentiment": "neutral", "confidence": 0.0, "scores": {}}
    classes = list(pipeline.classes_)
    probs = pipeline.predict_proba([cleaned])[0]
    top = int(probs.argmax())
    return {"sentiment": classes[top], "confidence": round(float(probs[top]), 4),
            "scores": {c: round(float(p), 4) for c, p in zip(classes, probs)}}
