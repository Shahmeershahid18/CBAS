"""
CRM-native Lead Scoring — training pipeline (app-native).

Trained on CBAS's OWN leads (exported by scripts/export-ai-training.ts to
data/crm_leads.csv) so it understands the CRM's real source/service vocabulary
("Trade Show", "Cold Call", "Website"...), which the Kaggle X-Education
benchmark model does not. Target = whether the lead converted.

Logistic Regression baseline -> XGBoost; best kept by ROC-AUC.

Run:  python -m training.lead_crm
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from app.config import ARTIFACTS_DIR, LEAD_CRM_ARTIFACT

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_CSV = DATA_DIR / "crm_leads.csv"

NUMERIC = ["quotation", "num_activities", "has_email", "has_phone"]
CATEGORICAL = ["source", "service"]
FEATURES = NUMERIC + CATEGORICAL
TARGET = "converted"


def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=0.01)),
    ])
    return ColumnTransformer([("num", numeric, NUMERIC), ("cat", categorical, CATEGORICAL)])


def _metrics(y_true, y_pred, y_prob) -> dict:
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_prob)), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def _importances(pipeline, top_n=12):
    pre: ColumnTransformer = pipeline.named_steps["pre"]
    names = list(pre.get_feature_names_out())
    model = pipeline.named_steps["model"]
    w = (np.asarray(model.feature_importances_, dtype=float)
         if hasattr(model, "feature_importances_")
         else np.abs(np.asarray(model.coef_, dtype=float).ravel()))
    order = np.argsort(w)[::-1][:top_n]
    return [{"feature": names[i].split("__", 1)[-1], "importance": round(float(w[i]), 4)}
            for i in order if w[i] > 0]


def train(csv_path: Path) -> dict:
    if not csv_path.exists():
        raise SystemExit(f"{csv_path} not found. Export it first:\n  npx tsx scripts/export-ai-training.ts")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df):,} leads from {csv_path.name} (conversion rate {df[TARGET].mean():.1%})")

    X, y = df[FEATURES], df[TARGET].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    baseline = Pipeline([("pre", build_preprocessor()),
                         ("model", LogisticRegression(max_iter=1000, class_weight="balanced"))])
    baseline.fit(X_train, y_train)
    b_prob = baseline.predict_proba(X_test)[:, 1]
    b_metrics = _metrics(y_test, (b_prob >= 0.5).astype(int), b_prob)
    print(f"Baseline  LogReg  -> F1 {b_metrics['f1']:.3f}  ROC-AUC {b_metrics['roc_auc']:.3f}")

    pos, neg = int((y_train == 1).sum()), int((y_train == 0).sum())
    improved = Pipeline([("pre", build_preprocessor()), ("model", XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.07, subsample=0.9, colsample_bytree=0.9,
        eval_metric="logloss", scale_pos_weight=(neg / pos if pos else 1.0), random_state=42, n_jobs=-1))])
    improved.fit(X_train, y_train)
    i_prob = improved.predict_proba(X_test)[:, 1]
    i_metrics = _metrics(y_test, (i_prob >= 0.5).astype(int), i_prob)
    print(f"Improved  XGBoost -> F1 {i_metrics['f1']:.3f}  ROC-AUC {i_metrics['roc_auc']:.3f}")

    if i_metrics["roc_auc"] >= b_metrics["roc_auc"]:
        best, name, metrics = improved, "xgboost", i_metrics
    else:
        best, name, metrics = baseline, "logistic_regression", b_metrics

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    bundle = {"pipeline": best, "model_name": name,
              "features": {"numeric": NUMERIC, "categorical": CATEGORICAL},
              "metrics": metrics, "baseline_metrics": b_metrics, "data_source": f"crm:{csv_path.name}",
              "top_features": _importances(best), "trained_rows": int(len(X_train))}
    joblib.dump(bundle, LEAD_CRM_ARTIFACT)
    (ARTIFACTS_DIR / "lead_crm_metrics.json").write_text(
        json.dumps({k: v for k, v in bundle.items() if k != "pipeline"}, indent=2))
    print(f"\nSaved best model ({name}) -> {LEAD_CRM_ARTIFACT}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the CBAS CRM-native lead-scoring model.")
    parser.add_argument("--data", type=str, default=None)
    args = parser.parse_args()
    train(Path(args.data) if args.data else DEFAULT_CSV)
