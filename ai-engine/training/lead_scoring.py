"""
Lead Scoring — training pipeline.

Follows the plan's Section 5 pipeline:
  1. Load + clean the dataset (Kaggle X-Education, or synthetic fallback)
  2. Encode categoricals / impute + scale numerics (sklearn ColumnTransformer)
  3. Stratified 80/20 train-test split
  4. Train a Logistic Regression baseline (interpretable)
  5. Train the improved XGBoost model
  6. Evaluate both on the held-out test set (Accuracy, Precision, Recall, F1, ROC-AUC)
  7. Save the better model (full preprocessing pipeline) with joblib

Run:
    python -m training.lead_scoring                # uses data/Lead Scoring.csv if present, else synthetic
    python -m training.lead_scoring --synthetic    # force synthetic
    python -m training.lead_scoring --data path/to/leads.csv
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

from app.config import ARTIFACTS_DIR, LEAD_SCORING_ARTIFACT
from app.models.lead_features import (
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    KAGGLE_COLUMN_MAP,
    KAGGLE_TARGET,
    NUMERIC_FEATURES,
    TARGET,
)
from training.synthetic import generate_leads

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_CSV = DATA_DIR / "Lead Scoring.csv"


def load_dataset(csv_path: Path | None, force_synthetic: bool) -> tuple[pd.DataFrame, str]:
    """Return (dataframe with internal column names + target, source_label)."""
    if not force_synthetic and csv_path and csv_path.exists():
        raw = pd.read_csv(csv_path)
        rename = {v: k for k, v in KAGGLE_COLUMN_MAP.items()}
        rename[KAGGLE_TARGET] = TARGET
        # Keep only the columns we model; tolerate a dataset missing some.
        present = {src: dst for src, dst in rename.items() if src in raw.columns}
        df = raw.rename(columns=present)
        missing = [c for c in ALL_FEATURES if c not in df.columns]
        for c in missing:
            df[c] = np.nan  # column absent in this CSV variant -> imputed later
        if TARGET not in df.columns:
            raise ValueError(f"Target column '{KAGGLE_TARGET}' not found in {csv_path}")
        df = df[ALL_FEATURES + [TARGET]].copy()
        # "Select" is X-Education's placeholder for "not answered" -> treat as missing.
        df = df.replace("Select", np.nan)
        return df, f"kaggle:{csv_path.name}"

    return generate_leads(), "synthetic"


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
    # Down-sample the ROC curve to ~100 points so the JSON stays small but the
    # figure is smooth.
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
    """Global importances from the fitted model, mapped back to readable names."""
    pre: ColumnTransformer = pipeline.named_steps["pre"]
    names = list(pre.get_feature_names_out())
    model = pipeline.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        weights = np.asarray(model.feature_importances_, dtype=float)
    else:  # logistic regression fallback
        weights = np.abs(np.asarray(model.coef_, dtype=float).ravel())
    order = np.argsort(weights)[::-1][:top_n]
    # Strip the ColumnTransformer prefixes ("num__", "cat__") for readability.
    return [
        {"feature": names[i].split("__", 1)[-1], "importance": round(float(weights[i]), 4)}
        for i in order if weights[i] > 0
    ]


def train(csv_path: Path | None, force_synthetic: bool) -> dict:
    df, source = load_dataset(csv_path, force_synthetic)
    print(f"Loaded {len(df):,} leads from {source} "
          f"(conversion rate {df[TARGET].mean():.1%})")

    X = df[ALL_FEATURES]
    y = df[TARGET].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pre = build_preprocessor()

    # 4. Baseline — Logistic Regression (interpretable)
    baseline = Pipeline([
        ("pre", build_preprocessor()),
        ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    baseline.fit(X_train, y_train)
    base_prob = baseline.predict_proba(X_test)[:, 1]
    base_metrics = _metrics(y_test, (base_prob >= 0.5).astype(int), base_prob)
    print(f"Baseline  LogReg  -> F1 {base_metrics['f1']:.3f}  ROC-AUC {base_metrics['roc_auc']:.3f}")

    # 5. Improved model — XGBoost
    pos = int((y_train == 1).sum())
    neg = int((y_train == 0).sum())
    scale_pos_weight = (neg / pos) if pos else 1.0
    improved = Pipeline([
        ("pre", pre),
        ("model", XGBClassifier(
            # Tuned via RandomizedSearchCV (scripts/tune_experiment.py).
            n_estimators=335,
            max_depth=3,
            learning_rate=0.0962,
            subsample=0.9446,
            colsample_bytree=0.7496,
            min_child_weight=4,
            gamma=0.0063,
            reg_lambda=0.5166,
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            n_jobs=-1,
        )),
    ])
    improved.fit(X_train, y_train)
    imp_prob = improved.predict_proba(X_test)[:, 1]
    imp_metrics = _metrics(y_test, (imp_prob >= 0.5).astype(int), imp_prob)
    print(f"Improved  XGBoost -> F1 {imp_metrics['f1']:.3f}  ROC-AUC {imp_metrics['roc_auc']:.3f}")

    # 6/7. Pick the better model by ROC-AUC and persist the whole pipeline.
    if imp_metrics["roc_auc"] >= base_metrics["roc_auc"]:
        best_pipeline, best_name, best_metrics = improved, "xgboost", imp_metrics
    else:
        best_pipeline, best_name, best_metrics = baseline, "logistic_regression", base_metrics

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
    joblib.dump(bundle, LEAD_SCORING_ARTIFACT)

    # Also drop a JSON metrics report — handy figures/tables for the report.
    report = {k: v for k, v in bundle.items() if k != "pipeline"}
    (ARTIFACTS_DIR / "lead_scoring_metrics.json").write_text(json.dumps(report, indent=2))

    print(f"\nSaved best model ({best_name}) -> {LEAD_SCORING_ARTIFACT}")
    print(f"Metrics report          -> {ARTIFACTS_DIR / 'lead_scoring_metrics.json'}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the CBAS lead-scoring model.")
    parser.add_argument("--data", type=str, default=None,
                        help="Path to a lead-scoring CSV (defaults to data/Lead Scoring.csv).")
    parser.add_argument("--synthetic", action="store_true",
                        help="Force the synthetic dataset even if a CSV is present.")
    args = parser.parse_args()

    csv = Path(args.data) if args.data else DEFAULT_CSV
    train(csv, force_synthetic=args.synthetic)
