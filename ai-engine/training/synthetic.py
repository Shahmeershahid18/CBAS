"""
Synthetic lead generator.

Lets the whole pipeline run end-to-end WITHOUT the Kaggle download, so the
service is demonstrable out of the box. The generated data has realistic,
learnable structure (engaged leads from good sources convert more often), but
it is NOT real: for the FYP defence, train on the real X-Education dataset
(see README) and treat this only as a fallback / smoke-test.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.models.lead_features import (
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET,
)

LEAD_ORIGINS = ["API", "Landing Page Submission", "Lead Add Form", "Lead Import"]
LEAD_SOURCES = ["Google", "Direct Traffic", "Organic Search", "Referral Sites",
                "Olark Chat", "Reference", "Welingak Website"]
LAST_ACTIVITIES = ["Email Opened", "SMS Sent", "Olark Chat Conversation",
                   "Page Visited on Website", "Converted to Lead",
                   "Email Bounced", "Form Submitted on Website"]
OCCUPATIONS = ["Unemployed", "Working Professional", "Student",
               "Businessman", "Housewife"]
SPECIALIZATIONS = ["Finance Management", "Marketing Management",
                   "Operations Management", "IT Projects Management",
                   "Business Administration", "Not Specified"]


def generate_leads(n: int = 9000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    total_visits = rng.poisson(3.5, n).astype(float)
    time_on_site = rng.gamma(2.0, 250.0, n)          # seconds
    page_views_per_visit = rng.gamma(2.0, 1.2, n)

    lead_source = rng.choice(LEAD_SOURCES, n)
    occupation = rng.choice(
        OCCUPATIONS, n, p=[0.55, 0.25, 0.10, 0.06, 0.04]
    )
    do_not_email = rng.choice(["No", "Yes"], n, p=[0.9, 0.1])

    df = pd.DataFrame({
        "total_visits": total_visits,
        "time_on_site": time_on_site,
        "page_views_per_visit": page_views_per_visit,
        "lead_origin": rng.choice(LEAD_ORIGINS, n),
        "lead_source": lead_source,
        "do_not_email": do_not_email,
        "last_activity": rng.choice(LAST_ACTIVITIES, n),
        "occupation": occupation,
        "specialization": rng.choice(SPECIALIZATIONS, n),
    })

    # Build a latent conversion propensity from a few signals, then sample the
    # binary target from it so the relationship is learnable but noisy.
    z = (
        -1.6
        + 0.0016 * df["time_on_site"]
        + 0.12 * df["total_visits"]
        + 0.15 * df["page_views_per_visit"]
        + np.where(df["occupation"] == "Working Professional", 1.4, 0.0)
        + np.where(df["lead_source"].isin(["Reference", "Welingak Website"]), 1.3, 0.0)
        + np.where(df["do_not_email"] == "Yes", -0.8, 0.0)
    )
    prob = 1.0 / (1.0 + np.exp(-z))
    df[TARGET] = (rng.random(n) < prob).astype(int)

    # Inject a little missingness so imputers earn their keep, like the real set.
    mask = rng.random(n) < 0.03
    df.loc[mask, "page_views_per_visit"] = np.nan

    return df[NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET]]
