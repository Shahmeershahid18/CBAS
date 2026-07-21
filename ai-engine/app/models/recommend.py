"""
Personalized Recommendations — inference wrapper.

Loads the joblib bundle from `training.recommend` and returns the top-N items
for a customer. Known customers get collaborative-filtering recommendations from
their latent vector (excluding items they already bought); unknown customers
fall back to the most popular items (a sensible cold-start default).
"""
from __future__ import annotations

import threading

import numpy as np

from app.config import RECOMMEND_ARTIFACT

_lock = threading.Lock()
_bundle: dict | None = None


class ModelNotTrainedError(RuntimeError):
    """Raised when no trained artifact exists yet."""


def _load() -> dict:
    global _bundle
    if _bundle is None:
        with _lock:
            if _bundle is None:
                if not RECOMMEND_ARTIFACT.exists():
                    raise ModelNotTrainedError(
                        f"No trained model at {RECOMMEND_ARTIFACT}. "
                        "Run: python -m training.recommend"
                    )
                import joblib
                _bundle = joblib.load(RECOMMEND_ARTIFACT)
    return _bundle


def is_ready() -> bool:
    return RECOMMEND_ARTIFACT.exists()


def model_info() -> dict:
    b = _load()
    return {
        "model_name": b["model_name"],
        "data_source": b["data_source"],
        "metrics": b["metrics"],
        "n_factors": b["n_factors"],
    }


def _item(bundle: dict, item_id: int) -> dict:
    return {
        "item_id": int(item_id),
        "name": bundle["item_names"].get(item_id, f"Item {item_id}"),
        "category": bundle["item_categories"].get(item_id) if bundle["item_categories"] else None,
    }


def _popular(bundle: dict, n: int, exclude: set[int]) -> list[dict]:
    pop = bundle["item_popularity"]
    order = np.argsort(-pop)
    out = [_item(bundle, i) for i in order if int(i) not in exclude]
    return out[:n]


def recommend(customer_id: str, n: int = 5) -> dict:
    """
    Return {customer_id, personalized, recommendations:[{item_id,name,category,score}]}.

    Uses ITEM-BASED collaborative filtering: each candidate service is scored by
    how similar it is (in SVD latent space) to the services THIS customer has
    already bought, so recommendations reflect the individual's own taste rather
    than global popularity. Unknown / no-purchase customers fall back to popularity.
    """
    bundle = _load()
    user_map = bundle["user_map"]
    internal = user_map.get(str(customer_id))
    seen = set(bundle["user_seen"].get(internal, [])) if internal is not None else set()

    if internal is None or not seen:
        recs = _popular(bundle, n, exclude=seen)
        for r in recs:
            r["score"] = None
        return {"customer_id": str(customer_id), "personalized": False, "recommendations": recs}

    # L2-normalize item vectors so a dot product is cosine similarity.
    item_factors = np.asarray(bundle["item_factors"], dtype=np.float32)
    norms = np.linalg.norm(item_factors, axis=1, keepdims=True)
    unit = item_factors / np.clip(norms, 1e-8, None)

    # Customer taste profile = centroid of the items they bought; score every
    # other item by cosine similarity to that profile.
    profile = unit[list(seen)].mean(axis=0)
    scores = unit @ profile
    for i in seen:
        scores[i] = -np.inf

    k = min(n, int(np.isfinite(scores).sum()))
    if k <= 0:
        recs = _popular(bundle, n, exclude=seen)
        for r in recs:
            r["score"] = None
        return {"customer_id": str(customer_id), "personalized": False, "recommendations": recs}

    top = np.argpartition(-scores, k - 1)[:k]
    top = top[np.argsort(-scores[top])]
    recs = []
    for i in top:
        item = _item(bundle, int(i))
        item["score"] = round(float(scores[i]), 4)
        recs.append(item)

    return {"customer_id": str(customer_id), "personalized": True, "recommendations": recs}
