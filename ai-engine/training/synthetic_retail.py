"""
Synthetic user-item purchase generator for recommendations. Builds interactions
with latent taste structure (users belong to segments that prefer certain
product categories) so collaborative filtering has real signal to learn.
Clearly synthetic: train on a real retail transactions set for the FYP defence.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CATEGORIES = {
    "Electronics": ["Wireless Mouse", "USB-C Hub", "Mechanical Keyboard", "Webcam 1080p",
                    "Noise-Cancelling Headphones", "Laptop Stand", "Portable SSD", "Monitor Arm"],
    "Office": ["Ergonomic Chair", "Standing Desk", "Desk Organizer", "Whiteboard",
               "Sticky Notes Pack", "Document Scanner", "Label Maker", "Filing Cabinet"],
    "Software": ["CRM License", "Antivirus 1yr", "VPN Subscription", "Cloud Storage 2TB",
                 "Design Suite", "Project Tool Seat", "Email Marketing Plan", "Analytics Add-on"],
    "Peripherals": ["HDMI Cable", "Power Bank", "Wireless Charger", "Bluetooth Speaker",
                    "Surge Protector", "Cable Sleeve", "Laptop Bag", "Screen Cleaner"],
}


def generate_interactions(n_users: int = 600, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (interactions[user_id,item_id,quantity], items[item_id,name,category])."""
    rng = np.random.default_rng(seed)

    items = []
    item_id = 0
    for cat, names in CATEGORIES.items():
        for name in names:
            items.append({"item_id": item_id, "name": name, "category": cat})
            item_id += 1
    items_df = pd.DataFrame(items)
    cats = list(CATEGORIES.keys())
    cat_of_item = items_df.set_index("item_id")["category"].to_dict()

    rows = []
    for user_id in range(n_users):
        # Each user has a primary + secondary preferred category.
        prefs = rng.choice(cats, size=2, replace=False)
        n_purchases = rng.integers(4, 16)
        for _ in range(n_purchases):
            # 75% from a preferred category, else random.
            if rng.random() < 0.75:
                cat = rng.choice(prefs)
                candidates = items_df[items_df["category"] == cat]["item_id"].values
            else:
                candidates = items_df["item_id"].values
            iid = int(rng.choice(candidates))
            rows.append({"user_id": user_id, "item_id": iid, "quantity": int(rng.integers(1, 4))})

    inter = (pd.DataFrame(rows)
             .groupby(["user_id", "item_id"], as_index=False)["quantity"].sum())
    return inter, items_df
