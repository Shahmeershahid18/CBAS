"""
Churn Prediction — training pipeline.

Same repeatable pipeline as lead scoring (plan Section 5), with a Random Forest
baseline and an XGBoost improved model — the two approaches the plan names for
churn. Trains on the Kaggle Telco Customer Churn dataset, or a synthetic
fallback.

Run:
    python -m training.churn                 # uses data/Telco-Customer-Churn.csv if present, else synthetic
    python -m training.churn --synthetic     # force synthetic
    python -m training.churn --data path/to/telco.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from app.config import ARTIFACTS_DIR, CHURN_ARTIFACT
from app.models.churn_features import (
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    KAGGLE_COLUMN_MAP,
    KAGGLE_TARGET,
    NUMERIC_FEATURES,
    TARGET,
)
from training.synthetic_churn import generate_customers

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_CSV = DATA_DIR / "Telco-Customer-Churn.csv"


def load_dataset(csv_path: Path | None, force_synthetic: bool) -> tuple[pd.DataFrame, str]:
    if not force_synthetic and csv_path and csv_path.exists():
        raw = pd.read_csv(csv_path)
        rename = {v: k for k, v in KAGGLE_COLUMN_MAP.items()}
        rename[KAGGLE_TARGET] = TARGET
        present = {src: dst for src, dst in rename.items() if src in raw.columns}
        df = raw.rename(columns=present)
        for c in ALL_FEATURES:
            if c not in df.columns:
                df[c] = np.nan
        if TARGET not in df.columns:
            raise ValueError(f"Target column '{KAGGLE_TARGET}' not found in {csv_path}")
        df = df[ALL_FEATURES + [TARGET]].copy()
        # Telco quirks: TotalCharges has blank strings; Churn is Yes/No.
        df["total_charges"] = pd.to_numeric(df["total_charges"], errors="coerce")
        df["senior_citizen"] = df["senior_citizen"].astype(str)
        if df[TARGET].dtype == object:
            df[TARGET] = (df[TARGET].astype(str).str.strip().str.lower() == "yes").astype(int)
        return df, f"kaggle:{csv_path.name}"

    return generate_customers(), "synthetic"


def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=0.01)),
    ])
    return ColumnTransformer([
        ("num", numeric, NUMERIC_FEATURES),
        ("cat", categorical, CATEGORICAL_FEATURES),
    ])


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
        "roc_curve": {"fpr": [round(float(x), 4) for x in fpr],
                      "tpr": [round(float(x), 4) for x in tpr]},
    }


def _feature_importances(pipeline: Pipeline, top_n: int = 12) -> list[dict]:
    pre: ColumnTransformer = pipeline.named_steps["pre"]
    names = list(pre.get_feature_names_out())
    model = pipeline.named_steps["model"]
    weights = np.asarray(model.feature_importances_, dtype=float)
    order = np.argsort(weights)[::-1][:top_n]
    return [
        {"feature": names[i].split("__", 1)[-1], "importance": round(float(weights[i]), 4)}
        for i in order if weights[i] > 0
    ]


def train(csv_path: Path | None, force_synthetic: bool) -> dict:
    df, source = load_dataset(csv_path, force_synthetic)
    print(f"Loaded {len(df):,} customers from {source} "
          f"(churn rate {df[TARGET].mean():.1%})")

    X = df[ALL_FEATURES]
    y = df[TARGET].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Baseline — Random Forest
    baseline = Pipeline([
        ("pre", build_preprocessor()),
        ("model", RandomForestClassifier(
            n_estimators=300, max_depth=12, class_weight="balanced",
            random_state=42, n_jobs=-1,
        )),
    ])
    baseline.fit(X_train, y_train)
    base_prob = baseline.predict_proba(X_test)[:, 1]
    base_metrics = _metrics(y_test, (base_prob >= 0.5).astype(int), base_prob)
    print(f"Baseline  RandomForest -> F1 {base_metrics['f1']:.3f}  ROC-AUC {base_metrics['roc_auc']:.3f}")

    # Improved — XGBoost
    pos = int((y_train == 1).sum())
    neg = int((y_train == 0).sum())
    scale_pos_weight = (neg / pos) if pos else 1.0
    improved = Pipeline([
        ("pre", build_preprocessor()),
        ("model", XGBClassifier(
            # Tuned via RandomizedSearchCV (scripts/tune_experiment.py).
            n_estimators=335, max_depth=3, learning_rate=0.0962,
            subsample=0.9446, colsample_bytree=0.7496, min_child_weight=4,
            gamma=0.0063, reg_lambda=0.5166, eval_metric="logloss",
            scale_pos_weight=scale_pos_weight, random_state=42, n_jobs=-1,
        )),
    ])
    improved.fit(X_train, y_train)
    imp_prob = improved.predict_proba(X_test)[:, 1]
    imp_metrics = _metrics(y_test, (imp_prob >= 0.5).astype(int), imp_prob)
    print(f"Improved  XGBoost      -> F1 {imp_metrics['f1']:.3f}  ROC-AUC {imp_metrics['roc_auc']:.3f}")

    if imp_metrics["roc_auc"] >= base_metrics["roc_auc"]:
        best_pipeline, best_name, best_metrics = improved, "xgboost", imp_metrics
    else:
        best_pipeline, best_name, best_metrics = baseline, "random_forest", base_metrics

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    bundle = {
        "pipeline": best_pipeline,
        "model_name": best_name,
        "features": {"numeric": NUMERIC_FEATURES, "categorical": CATEGORICAL_FEATURES},
        "metrics": best_metrics,
        "baseline_metrics": base_metrics,
        "data_source": source,
        "top_features": _feature_importances(best_pipeline),
        "trained_rows": int(len(X_train)),
    }
    joblib.dump(bundle, CHURN_ARTIFACT)

    report = {k: v for k, v in bundle.items() if k != "pipeline"}
    (ARTIFACTS_DIR / "churn_metrics.json").write_text(json.dumps(report, indent=2))

    print(f"\nSaved best model ({best_name}) -> {CHURN_ARTIFACT}")
    print(f"Metrics report          -> {ARTIFACTS_DIR / 'churn_metrics.json'}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the CBAS churn model.")
    parser.add_argument("--data", type=str, default=None,
                        help="Path to a Telco churn CSV (defaults to data/Telco-Customer-Churn.csv).")
    parser.add_argument("--synthetic", action="store_true",
                        help="Force the synthetic dataset even if a CSV is present.")
    args = parser.parse_args()

    csv = Path(args.data) if args.data else DEFAULT_CSV
    train(csv, force_synthetic=args.synthetic)
