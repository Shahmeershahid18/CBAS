"""
Sentiment Analysis — training pipeline.

Per the plan: text cleaning + TF-IDF vectorization + Logistic Regression — the
fast, light, easy-to-defend baseline. Three classes: positive / negative /
neutral. Trains on a labelled review CSV or a synthetic fallback.

Accepted CSV: any file with a text column (text / review / Review / content /
Text) and a label column (sentiment / label / Sentiment / rating / Score).
Numeric ratings are mapped 1-2 -> negative, 3 -> neutral, 4-5 -> positive.

Run:
    python -m training.sentiment                     # data/reviews.csv if present, else synthetic
    python -m training.sentiment --synthetic
    python -m training.sentiment --data path/to/reviews.csv
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from app.config import ARTIFACTS_DIR, SENTIMENT_ARTIFACT
from training.synthetic_sentiment import generate_reviews

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_CSV = DATA_DIR / "reviews.csv"

TEXT_COLS = ["text", "review", "Review", "content", "Text", "reviewText", "verified_reviews"]
LABEL_COLS = ["sentiment", "label", "Sentiment", "Label", "rating", "Score", "stars", "feedback"]

LABELS = ["negative", "neutral", "positive"]

_clean_re = re.compile(r"[^a-z0-9\s]")


def clean_text(s: str) -> str:
    s = str(s).lower()
    s = _clean_re.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip()


def _map_label(v) -> str | None:
    """Normalise a raw label/rating value onto negative/neutral/positive."""
    if pd.isna(v):
        return None
    # numeric rating?
    try:
        num = float(v)
        if num <= 2:
            return "negative"
        if num == 3:
            return "neutral"
        if num >= 4:
            return "positive"
    except (ValueError, TypeError):
        pass
    t = str(v).strip().lower()
    if t in ("positive", "pos", "1", "good", "happy"):
        return "positive"
    if t in ("negative", "neg", "0", "-1", "bad", "sad"):
        return "negative"
    if t in ("neutral", "neu", "average"):
        return "neutral"
    return None


def load_dataset(csv_path: Path | None, force_synthetic: bool) -> tuple[pd.DataFrame, str]:
    if not force_synthetic and csv_path and csv_path.exists():
        # Review dumps are often not UTF-8; fall back to latin-1 rather than crash.
        try:
            raw = pd.read_csv(csv_path)
        except UnicodeDecodeError:
            raw = pd.read_csv(csv_path, encoding="latin-1")
        text_col = next((c for c in TEXT_COLS if c in raw.columns), None)
        label_col = next((c for c in LABEL_COLS if c in raw.columns), None)
        if not text_col or not label_col:
            raise ValueError(
                f"Could not find text/label columns in {csv_path}. "
                f"Looked for text in {TEXT_COLS} and label in {LABEL_COLS}."
            )
        df = pd.DataFrame({
            "text": raw[text_col].astype(str),
            "sentiment": raw[label_col].map(_map_label),
        }).dropna(subset=["sentiment"])
        return df, f"kaggle:{csv_path.name}"

    return generate_reviews(), "synthetic"


def train(csv_path: Path | None, force_synthetic: bool) -> dict:
    df, source = load_dataset(csv_path, force_synthetic)
    df["text"] = df["text"].map(clean_text)
    df = df[df["text"].str.len() > 0]
    dist = df["sentiment"].value_counts().to_dict()
    print(f"Loaded {len(df):,} labelled texts from {source}. Class distribution: {dist}")

    X = df["text"]
    y = df["sentiment"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # TF-IDF + Logistic Regression (the plan's baseline).
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2), min_df=2, max_df=0.9, sublinear_tf=True,
            stop_words="english",
        )),
        ("model", LogisticRegression(max_iter=1000, class_weight="balanced", C=4.0)),
    ])
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    metrics = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_test, y_pred, average="macro")), 4),
        "per_class": classification_report(y_test, y_pred, output_dict=True, zero_division=0),
        "labels": sorted(y.unique().tolist()),
        "confusion_matrix": confusion_matrix(
            y_test, y_pred, labels=sorted(y.unique().tolist())
        ).tolist(),
    }
    print(f"TF-IDF + LogReg -> accuracy {metrics['accuracy']:.3f}  macro-F1 {metrics['macro_f1']:.3f}")

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    bundle = {
        "pipeline": pipeline,
        "model_name": "tfidf_logreg",
        "labels": sorted(y.unique().tolist()),
        "metrics": metrics,
        "data_source": source,
        "trained_rows": int(len(X_train)),
    }
    joblib.dump(bundle, SENTIMENT_ARTIFACT)

    report = {k: v for k, v in bundle.items() if k != "pipeline"}
    (ARTIFACTS_DIR / "sentiment_metrics.json").write_text(json.dumps(report, indent=2))

    print(f"\nSaved model (tfidf_logreg) -> {SENTIMENT_ARTIFACT}")
    print(f"Metrics report            -> {ARTIFACTS_DIR / 'sentiment_metrics.json'}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the CBAS sentiment model.")
    parser.add_argument("--data", type=str, default=None,
                        help="Path to a labelled review CSV (defaults to data/reviews.csv).")
    parser.add_argument("--synthetic", action="store_true",
                        help="Force the synthetic dataset even if a CSV is present.")
    args = parser.parse_args()

    csv = Path(args.data) if args.data else DEFAULT_CSV
    train(csv, force_synthetic=args.synthetic)
