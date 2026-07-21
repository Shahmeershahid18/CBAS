"""
CRM-note Sentiment — training pipeline (app-native).

Same TF-IDF + Logistic Regression approach as the review sentiment model, but
trained on CRM/sales business language so it classifies activity notes correctly
(the review-trained model mislabels them). Kept separate; the review model
remains the benchmark deliverable.

Run:  python -m training.sentiment_crm
"""
from __future__ import annotations

import json

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from app.config import ARTIFACTS_DIR, SENTIMENT_CRM_ARTIFACT
from training.sentiment import clean_text
from training.synthetic_crm_sentiment import generate_crm_notes


def train() -> dict:
    df = generate_crm_notes()
    df["text"] = df["text"].map(clean_text)
    print(f"Loaded {len(df):,} CRM notes. Classes: {df['sentiment'].value_counts().to_dict()}")

    X, y = df["text"], df["sentiment"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
        ("model", LogisticRegression(max_iter=1000, class_weight="balanced", C=6.0)),
    ])
    pipeline.fit(X_tr, y_tr)
    y_pred = pipeline.predict(X_te)
    labels = sorted(y.unique().tolist())
    metrics = {
        "accuracy": round(float(accuracy_score(y_te, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_te, y_pred, average="macro")), 4),
        "per_class": classification_report(y_te, y_pred, output_dict=True, zero_division=0),
        "labels": labels,
        "confusion_matrix": confusion_matrix(y_te, y_pred, labels=labels).tolist(),
    }
    print(f"TF-IDF + LogReg (CRM notes) -> accuracy {metrics['accuracy']:.3f}  macro-F1 {metrics['macro_f1']:.3f}")

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    bundle = {"pipeline": pipeline, "model_name": "tfidf_logreg_crm", "labels": labels,
              "metrics": metrics, "data_source": "synthetic_crm_notes", "trained_rows": int(len(X_tr))}
    joblib.dump(bundle, SENTIMENT_CRM_ARTIFACT)
    (ARTIFACTS_DIR / "sentiment_crm_metrics.json").write_text(
        json.dumps({k: v for k, v in bundle.items() if k != "pipeline"}, indent=2))
    print(f"Saved model -> {SENTIMENT_CRM_ARTIFACT}")
    return metrics


if __name__ == "__main__":
    train()
