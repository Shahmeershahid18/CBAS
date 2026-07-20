"""
CRM-native Churn — training pipeline.

Unlike the Telco benchmark model (training/churn.py), this model is trained on
CBAS's OWN data, exported by scripts/export-ai-training.ts to data/crm_churn.csv.
A customer is labelled "lapsed" when their most recent engagement is older than
the lapse window; the model predicts that from engagement signals (orders,
spend, sentiment, deal state, tenure) — the recency that defines the label is
deliberately excluded from the features.

Random Forest baseline vs. XGBoost; best kept by ROC-AUC.

Run:  python -m training.churn_crm
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from app.config import ARTIFACTS_DIR, CHURN_CRM_ARTIFACT

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_CSV = DATA_DIR / "crm_churn.csv"

FEATURES = [
    "tenure_days", "num_orders", "total_spend", "num_activities",
    "num_negative_notes", "avg_sentiment", "num_categories", "has_open_deal",
]
TARGET = "churned"


def _metrics(y_true, y_pred, y_prob) -> dict:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    if len(fpr) > 100:
        idx = np.linspace(0, len(fpr) - 1, 100).astype(int)
        fpr, tpr = fpr[idx], tpr[idx]
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_prob)), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "roc_curve": {"fpr": [round(float(x), 4) for x in fpr], "tpr": [round(float(x), 4) for x in tpr]},
    }


def _importances(pipeline, top_n=8):
    model = pipeline.named_steps["model"]
    w = np.asarray(model.feature_importances_, dtype=float)
    order = np.argsort(w)[::-1][:top_n]
    return [{"feature": FEATURES[i], "importance": round(float(w[i]), 4)} for i in order if w[i] > 0]


def train(csv_path: Path) -> dict:
    if not csv_path.exists():
        raise SystemExit(f"{csv_path} not found. Export it first:\n"
                         "  npx tsx scripts/export-ai-training.ts")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df):,} contacts from {csv_path.name} (churn rate {df[TARGET].mean():.1%})")

    X, y = df[FEATURES], df[TARGET].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    def pre():
        return Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])

    baseline = Pipeline([("pre", pre()), ("model", RandomForestClassifier(
        n_estimators=300, max_depth=8, class_weight="balanced", random_state=42, n_jobs=-1))])
    baseline.fit(X_train, y_train)
    b_prob = baseline.predict_proba(X_test)[:, 1]
    b_metrics = _metrics(y_test, (b_prob >= 0.5).astype(int), b_prob)
    print(f"Baseline  RandomForest -> F1 {b_metrics['f1']:.3f}  ROC-AUC {b_metrics['roc_auc']:.3f}")

    pos, neg = int((y_train == 1).sum()), int((y_train == 0).sum())
    improved = Pipeline([("pre", pre()), ("model", XGBClassifier(
        n_estimators=300, max_depth=3, learning_rate=0.08, subsample=0.9, colsample_bytree=0.9,
        eval_metric="logloss", scale_pos_weight=(neg / pos if pos else 1.0), random_state=42, n_jobs=-1))])
    improved.fit(X_train, y_train)
    i_prob = improved.predict_proba(X_test)[:, 1]
    i_metrics = _metrics(y_test, (i_prob >= 0.5).astype(int), i_prob)
    print(f"Improved  XGBoost      -> F1 {i_metrics['f1']:.3f}  ROC-AUC {i_metrics['roc_auc']:.3f}")

    if i_metrics["roc_auc"] >= b_metrics["roc_auc"]:
        best, name, metrics = improved, "xgboost", i_metrics
    else:
        best, name, metrics = baseline, "random_forest", b_metrics

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    bundle = {
        "pipeline": best, "model_name": name, "features": FEATURES, "metrics": metrics,
        "baseline_metrics": b_metrics, "data_source": f"crm:{csv_path.name}",
        "top_features": _importances(best), "trained_rows": int(len(X_train)),
    }
    joblib.dump(bundle, CHURN_CRM_ARTIFACT)
    report = {k: v for k, v in bundle.items() if k != "pipeline"}
    (ARTIFACTS_DIR / "churn_crm_metrics.json").write_text(json.dumps(report, indent=2))
    print(f"\nSaved best model ({name}) -> {CHURN_CRM_ARTIFACT}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the CBAS CRM-native churn model.")
    parser.add_argument("--data", type=str, default=None)
    args = parser.parse_args()
    train(Path(args.data) if args.data else DEFAULT_CSV)
