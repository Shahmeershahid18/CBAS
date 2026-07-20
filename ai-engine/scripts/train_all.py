"""
Retrain all four models in one go. Each trainer uses the real CSV in data/ if
present, otherwise falls back to synthetic (and says so).

Run:
    python -m scripts.train_all
"""
from __future__ import annotations

from training import churn, lead_scoring, recommend, sentiment
from training.churn import DEFAULT_CSV as CHURN_CSV
from training.lead_scoring import DEFAULT_CSV as LEAD_CSV
from training.recommend import DEFAULT_CSV as RECOMMEND_CSV
from training.sentiment import DEFAULT_CSV as SENTIMENT_CSV


def main() -> None:
    jobs = [
        ("Lead Scoring", lead_scoring.train, LEAD_CSV),
        ("Churn",        churn.train,        CHURN_CSV),
        ("Sentiment",    sentiment.train,    SENTIMENT_CSV),
        ("Recommendations", recommend.train, RECOMMEND_CSV),
    ]
    for name, train_fn, csv in jobs:
        print("\n" + "=" * 70)
        print(f"Training: {name}  ({'real CSV' if csv.exists() else 'synthetic fallback'})")
        print("=" * 70)
        train_fn(csv, force_synthetic=False)

    print("\nAll models retrained. Restart the API to load them:")
    print("  uvicorn app.main:app --port 8088")


if __name__ == "__main__":
    main()
