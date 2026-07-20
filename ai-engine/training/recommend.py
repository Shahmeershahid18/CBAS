"""
Personalized Recommendations — training pipeline.

Collaborative filtering via matrix factorization. Builds a user-item interaction
matrix from purchase history and factorizes it with Truncated SVD (the same
latent-factor idea behind `surprise`/`implicit`, using only scipy/scikit-learn
so it installs cleanly on Windows). Learns user and item latent vectors; a
user's score for an item is the dot product of their vectors.

Evaluated with Precision@K / Recall@K on held-out interactions (leave-out).

Accepted CSV: retail transactions with a user column (CustomerID / user_id /
UserId), an item column (StockCode / item_id / ProductId / Description) and
optionally a quantity column. An items CSV (item_id,name) is optional.

Run:
    python -m training.recommend                  # data/retail.csv if present, else synthetic
    python -m training.recommend --synthetic
    python -m training.recommend --data path/to/retail.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

from app.config import ARTIFACTS_DIR, RECOMMEND_ARTIFACT
from training.synthetic_retail import generate_interactions

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_CSV = DATA_DIR / "retail.csv"

USER_COLS = ["user_id", "CustomerID", "UserId", "customer_id", "Customer Id"]
ITEM_COLS = ["item_id", "StockCode", "ProductId", "product_id", "Description", "item"]
NAME_COLS = ["item_name", "Description", "ProductName", "product_name", "name", "Product Name"]
QTY_COLS = ["quantity", "Quantity", "qty"]

N_FACTORS = 32
TOP_K = 5


def load_dataset(csv_path: Path | None, force_synthetic: bool):
    """Returns (interactions[user_id,item_id,quantity], items_df[item_id,name], source)."""
    if not force_synthetic and csv_path and csv_path.exists():
        raw = pd.read_csv(csv_path, encoding="latin-1")
        ucol = next((c for c in USER_COLS if c in raw.columns), None)
        icol = next((c for c in ITEM_COLS if c in raw.columns), None)
        if not ucol or not icol:
            raise ValueError(f"Could not find user/item columns in {csv_path}.")
        qcol = next((c for c in QTY_COLS if c in raw.columns), None)
        # A friendlier item label column (e.g. Description) when the item key is a code.
        ncol = next((c for c in NAME_COLS if c in raw.columns and c != icol), None)
        cols = [ucol, icol] + ([ncol] if ncol else []) + ([qcol] if qcol else [])
        df = raw[cols].dropna(subset=[ucol, icol]).copy()
        df.columns = ["user_raw", "item_raw"] + (["item_name"] if ncol else []) + (["quantity"] if qcol else [])

        # Normalize ids to clean strings: float customer ids like 17850.0 -> "17850"
        # so the app/API can query by the natural id.
        df["user_raw"] = df["user_raw"].astype(str).str.replace(r"\.0$", "", regex=True)
        df["item_raw"] = df["item_raw"].astype(str)
        if not ncol:
            df["item_name"] = df["item_raw"]
        else:
            df["item_name"] = df["item_name"].astype(str)
        if qcol:
            df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce").fillna(1).clip(lower=1)
        else:
            df["quantity"] = 1

        # Map raw ids to contiguous ints.
        df["user_id"] = df["user_raw"].astype("category").cat.codes
        df["item_id"] = df["item_raw"].astype("category").cat.codes
        # Item label = the most common description for that item code.
        items_df = (df.groupby("item_id")["item_name"]
                    .agg(lambda s: s.mode().iat[0] if not s.mode().empty else s.iat[0])
                    .reset_index().rename(columns={"item_name": "name"}))
        inter = df.groupby(["user_id", "item_id"], as_index=False)["quantity"].sum()
        # Preserve the raw customer id -> internal id mapping so the app can query by real id.
        user_map = (df[["user_raw", "user_id"]].drop_duplicates()
                    .set_index("user_raw")["user_id"].astype(int).to_dict())
        return inter, items_df, {str(k): v for k, v in user_map.items()}, f"kaggle:{csv_path.name}"

    inter, items_df = generate_interactions()
    user_map = {str(u): int(u) for u in inter["user_id"].unique()}
    return inter, items_df, user_map, "synthetic"


def _build_matrix(inter: pd.DataFrame, n_users: int, n_items: int) -> csr_matrix:
    # Implicit feedback: log1p of quantity dampens heavy buyers.
    data = np.log1p(inter["quantity"].to_numpy(dtype=float))
    return csr_matrix((data, (inter["user_id"], inter["item_id"])), shape=(n_users, n_items))


def _precision_recall_at_k(user_factors, item_factors, train_mat, test_by_user, k=TOP_K):
    precisions, recalls = [], []
    for u, held in test_by_user.items():
        if not held:
            continue
        scores = item_factors @ user_factors[u]
        # Exclude items already seen in training.
        seen = train_mat[u].indices
        scores[seen] = -np.inf
        top = np.argpartition(-scores, k)[:k]
        top = top[np.argsort(-scores[top])]
        hits = len(set(top.tolist()) & held)
        precisions.append(hits / k)
        recalls.append(hits / len(held))
    return (round(float(np.mean(precisions)), 4) if precisions else 0.0,
            round(float(np.mean(recalls)), 4) if recalls else 0.0)


def train(csv_path: Path | None, force_synthetic: bool) -> dict:
    inter, items_df, user_map, source = load_dataset(csv_path, force_synthetic)
    n_users = int(inter["user_id"].max()) + 1
    n_items = int(inter["item_id"].max()) + 1
    print(f"Loaded {len(inter):,} interactions from {source}: "
          f"{n_users:,} users x {n_items:,} items")

    # Hold out one interaction per eligible user for Precision@K / Recall@K.
    rng = np.random.default_rng(42)
    test_by_user: dict[int, set] = {}
    train_rows = []
    for u, grp in inter.groupby("user_id"):
        idx = grp.index.to_numpy()
        if len(idx) >= 4:
            held = rng.choice(idx, size=max(1, len(idx) // 5), replace=False)
            test_by_user[int(u)] = set(inter.loc[held, "item_id"].astype(int).tolist())
            train_rows.append(grp.drop(index=held))
        else:
            train_rows.append(grp)
    train_inter = pd.concat(train_rows)

    train_mat = _build_matrix(train_inter, n_users, n_items)

    n_factors = min(N_FACTORS, n_items - 1, n_users - 1)
    svd = TruncatedSVD(n_components=n_factors, random_state=42)
    user_factors = svd.fit_transform(train_mat)          # users x factors
    item_factors = svd.components_.T                     # items x factors

    p_at_k, r_at_k = _precision_recall_at_k(user_factors, item_factors, train_mat, test_by_user)
    print(f"TruncatedSVD ({n_factors} factors) -> Precision@{TOP_K} {p_at_k:.3f}  Recall@{TOP_K} {r_at_k:.3f}")

    # Full-data factors for serving (retrain on everything).
    full_mat = _build_matrix(inter, n_users, n_items)
    user_factors_full = svd.fit_transform(full_mat)
    item_factors_full = svd.components_.T
    item_popularity = np.asarray(full_mat.sum(axis=0)).ravel()

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model_name": "truncated_svd_cf",
        "user_factors": user_factors_full.astype(np.float32),
        "item_factors": item_factors_full.astype(np.float32),
        "item_popularity": item_popularity.astype(np.float32),
        "user_seen": {int(u): full_mat[u].indices.tolist() for u in range(n_users)},
        "item_names": items_df.set_index("item_id")["name"].to_dict(),
        "item_categories": (items_df.set_index("item_id")["category"].to_dict()
                            if "category" in items_df.columns else {}),
        "user_map": user_map,                 # raw customer id (str) -> internal row
        "n_factors": n_factors,
        "metrics": {f"precision_at_{TOP_K}": p_at_k, f"recall_at_{TOP_K}": r_at_k,
                    "n_users": n_users, "n_items": n_items},
        "data_source": source,
    }
    joblib.dump(bundle, RECOMMEND_ARTIFACT)

    report = {"model_name": bundle["model_name"], "metrics": bundle["metrics"],
              "data_source": source, "n_factors": n_factors}
    (ARTIFACTS_DIR / "recommend_metrics.json").write_text(json.dumps(report, indent=2))

    print(f"\nSaved model (truncated_svd_cf) -> {RECOMMEND_ARTIFACT}")
    print(f"Metrics report                -> {ARTIFACTS_DIR / 'recommend_metrics.json'}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the CBAS recommendation model.")
    parser.add_argument("--data", type=str, default=None,
                        help="Path to a retail transactions CSV (defaults to data/retail.csv).")
    parser.add_argument("--synthetic", action="store_true",
                        help="Force the synthetic dataset even if a CSV is present.")
    args = parser.parse_args()

    csv = Path(args.data) if args.data else DEFAULT_CSV
    train(csv, force_synthetic=args.synthetic)
