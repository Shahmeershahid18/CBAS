"""
Sentiment Analysis — inference wrapper.

Loads the joblib bundle from `training.sentiment` and labels a piece of text
positive / negative / neutral with a confidence score.
"""
from __future__ import annotations

import re
import threading

import joblib

from app.config import SENTIMENT_ARTIFACT

_lock = threading.Lock()
_bundle: dict | None = None
_clean_re = re.compile(r"[^a-z0-9\s]")


class ModelNotTrainedError(RuntimeError):
    """Raised when no trained artifact exists yet."""


def _clean(s: str) -> str:
    s = _clean_re.sub(" ", str(s).lower())
    return re.sub(r"\s+", " ", s).strip()


def _load() -> dict:
    global _bundle
    if _bundle is None:
        with _lock:
            if _bundle is None:
                if not SENTIMENT_ARTIFACT.exists():
                    raise ModelNotTrainedError(
                        f"No trained model at {SENTIMENT_ARTIFACT}. "
                        "Run: python -m training.sentiment"
                    )
                _bundle = joblib.load(SENTIMENT_ARTIFACT)
    return _bundle


def is_ready() -> bool:
    return SENTIMENT_ARTIFACT.exists()


def model_info() -> dict:
    b = _load()
    return {
        "model_name": b["model_name"],
        "data_source": b["data_source"],
        "labels": b["labels"],
        "metrics": {"accuracy": b["metrics"]["accuracy"], "macro_f1": b["metrics"]["macro_f1"]},
        "trained_rows": b["trained_rows"],
    }


def analyze_sentiment(text: str) -> dict:
    """Returns {sentiment, confidence, scores} for a single text."""
    bundle = _load()
    pipeline = bundle["pipeline"]

    cleaned = _clean(text)
    if not cleaned:
        return {"sentiment": "neutral", "confidence": 0.0, "scores": {}}

    classes = list(pipeline.classes_)
    probs = pipeline.predict_proba([cleaned])[0]
    top_idx = int(probs.argmax())
    scores = {cls: round(float(p), 4) for cls, p in zip(classes, probs)}

    return {
        "sentiment": classes[top_idx],
        "confidence": round(float(probs[top_idx]), 4),
        "scores": scores,
    }
