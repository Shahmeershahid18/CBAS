"""
Synthetic labelled-review generator for sentiment — lets the pipeline run
without a download. Three classes (positive / negative / neutral) built from
templated phrases. Clearly synthetic: train on real review data (Amazon /
product reviews) for the FYP defence.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

POSITIVE = [
    "absolutely love this product, works perfectly",
    "excellent service and fast delivery, highly recommend",
    "great quality, exceeded my expectations",
    "fantastic experience, the team was very helpful",
    "amazing value for money, very satisfied",
    "the support was outstanding and quick to respond",
    "best purchase I have made this year, wonderful",
    "super easy to use and reliable, happy customer",
    "brilliant product, five stars from me",
    "really impressed with the quality and speed",
]
NEGATIVE = [
    "terrible experience, the product broke immediately",
    "very disappointed, poor quality and slow delivery",
    "awful customer service, no one responded to me",
    "waste of money, it stopped working after a day",
    "horrible, would not recommend to anyone",
    "the worst support I have ever dealt with",
    "completely useless, full of bugs and errors",
    "frustrating and unreliable, want a refund",
    "cheap materials, fell apart within a week",
    "extremely unhappy with this purchase, avoid",
]
NEUTRAL = [
    "the product arrived on time as described",
    "it is okay, does what it says nothing special",
    "average experience, neither good nor bad",
    "the item works fine, standard quality",
    "delivery was normal, product is acceptable",
    "it is fine for the price, no strong opinion",
    "received the order, looks like the picture",
    "does the job, not remarkable in any way",
    "reasonable product, meets basic expectations",
    "no major issues, just an ordinary experience",
]

FILLERS = ["", " Thanks.", " Overall.", " Just my opinion.", " For reference.",
           " Ordered last week.", " Will see how it goes.", " As expected."]


def generate_reviews(n: int = 9000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    banks = {"positive": POSITIVE, "negative": NEGATIVE, "neutral": NEUTRAL}
    labels = rng.choice(["positive", "negative", "neutral"], n)

    texts = []
    for lab in labels:
        base = rng.choice(banks[lab])
        filler = rng.choice(FILLERS)
        texts.append(base + filler)

    return pd.DataFrame({"text": texts, "sentiment": labels})
