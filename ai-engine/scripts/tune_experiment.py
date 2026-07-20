"""
Quick hyperparameter-search experiment for the two classification models, to see
whether accuracy/ROC-AUC can be improved beyond the current hand-set values.
Prints current-vs-tuned CV ROC-AUC and the best params found. Does NOT overwrite
the served models — it only reports, so we decide whether tuning is worth baking in.

Run:  python -m scripts.tune_experiment
"""
from __future__ import annotations

import numpy as np
from scipy.stats import randint, uniform
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from training import churn as churn_mod
from training import lead_scoring as lead_mod


def run(name, load_dataset, default_csv, build_pre, all_features, target):
    df, source = load_dataset(default_csv, force_synthetic=False)
    X = df[all_features]
    y = df[target].astype(int)
    print(f"\n=== {name} ({source}, {len(df):,} rows) ===")

    neg, pos = int((y == 0).sum()), int((y == 1).sum())
    spw = neg / pos if pos else 1.0

    # Current configuration (what the trainer ships today).
    current = Pipeline([
        ("pre", build_pre()),
        ("model", XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.08,
                                subsample=0.9, colsample_bytree=0.9, eval_metric="logloss",
                                scale_pos_weight=spw, random_state=42, n_jobs=-1)),
    ])
    cur_auc = cross_val_score(current, X, y, cv=3, scoring="roc_auc", n_jobs=-1).mean()
    print(f"current  CV ROC-AUC: {cur_auc:.4f}")

    # Randomized search over a sensible XGBoost space.
    search_pipe = Pipeline([
        ("pre", build_pre()),
        ("model", XGBClassifier(eval_metric="logloss", scale_pos_weight=spw,
                                random_state=42, n_jobs=-1)),
    ])
    param_dist = {
        "model__n_estimators": randint(200, 600),
        "model__max_depth": randint(3, 8),
        "model__learning_rate": uniform(0.02, 0.18),
        "model__subsample": uniform(0.7, 0.3),
        "model__colsample_bytree": uniform(0.7, 0.3),
        "model__min_child_weight": randint(1, 8),
        "model__gamma": uniform(0, 0.4),
        "model__reg_lambda": uniform(0.5, 3.0),
    }
    search = RandomizedSearchCV(
        search_pipe, param_dist, n_iter=30, cv=3, scoring="roc_auc",
        random_state=42, n_jobs=-1, verbose=0,
    )
    search.fit(X, y)
    print(f"tuned    CV ROC-AUC: {search.best_score_:.4f}  (delta {search.best_score_ - cur_auc:+.4f})")
    best = {k.replace('model__', ''): (round(v, 4) if isinstance(v, float) else v)
            for k, v in search.best_params_.items()}
    print(f"best params: {best}")


if __name__ == "__main__":
    run("Lead Scoring", lead_mod.load_dataset, lead_mod.DEFAULT_CSV,
        lead_mod.build_preprocessor, lead_mod.ALL_FEATURES, lead_mod.TARGET)
    run("Churn", churn_mod.load_dataset, churn_mod.DEFAULT_CSV,
        churn_mod.build_preprocessor, churn_mod.ALL_FEATURES, churn_mod.TARGET)
