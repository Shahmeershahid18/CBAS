"""
Generator for CRM-note sentiment training data.

The public review model mislabels CRM notes ("signed the contract", "renewed",
"thrilled") because it never saw business language. This builds labelled
sales/CRM notes from a broad business vocabulary and varied sentence templates,
so a TF-IDF model learns the words that actually carry sentiment in a CRM.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SUBJECTS = ["client", "customer", "account", "lead", "prospect", "buyer",
            "stakeholder", "contact", "decision maker", "the team"]

POS_PHRASES = [
    "was thrilled with the demo", "signed the contract today", "renewed early",
    "is very happy with our service", "loved the proposal", "upgraded their plan",
    "gave great feedback", "closed the deal", "is excited to expand usage",
    "was delighted with the results", "recommended us to a partner",
    "approved the budget", "is impressed with the onboarding",
    "confirmed the purchase", "expanded to more seats", "praised the support team",
    "agreed to a long-term contract", "is satisfied and staying",
    "responded very positively", "wants to buy more",
]
NEG_PHRASES = [
    "is angry about the delays", "is threatening to cancel", "filed a complaint",
    "is very unhappy with the service", "wants a refund", "is frustrated with the bugs",
    "escalated the issue", "is disappointed with the results", "churned this month",
    "reported a serious problem", "was upset about the pricing",
    "is considering a competitor", "left negative feedback", "is dissatisfied",
    "cancelled the subscription", "complained about slow support",
    "is unresponsive and disengaged", "rejected the proposal",
    "had a bad experience", "downgraded due to issues",
]
NEU_PHRASES = [
    "called and left a voicemail", "awaiting their reply", "scheduled a follow-up",
    "sent the proposal for review", "requested more information", "is reviewing the quote",
    "had a routine check-in call", "discussed the roadmap", "no update this week",
    "emailed the contract for signature", "is comparing options", "asked about pricing",
    "meeting set for next week", "shared the documentation", "is evaluating the trial",
    "confirmed receipt of the invoice", "pending internal approval",
    "we followed up as planned", "updated their contact details", "attended the webinar",
]

TEMPLATES = [
    "The {subj} {phrase}.",
    "{Subj} {phrase}.",
    "Note: {subj} {phrase}.",
    "Just spoke with the {subj} — {phrase}.",
    "Update: {subj} {phrase}.",
    "Follow-up: the {subj} {phrase}.",
    "{Subj} {phrase} after the call.",
]


def _rows(rng, phrases, label, n):
    out = []
    for _ in range(n):
        subj = rng.choice(SUBJECTS)
        phrase = rng.choice(phrases)
        tmpl = rng.choice(TEMPLATES)
        text = tmpl.format(subj=subj, Subj=subj.capitalize(), phrase=phrase)
        out.append({"text": text, "sentiment": label})
    return out


def generate_crm_notes(n_per_class: int = 2500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = (_rows(rng, POS_PHRASES, "positive", n_per_class)
            + _rows(rng, NEG_PHRASES, "negative", n_per_class)
            + _rows(rng, NEU_PHRASES, "neutral", n_per_class))
    return pd.DataFrame(rows).sample(frac=1, random_state=seed).reset_index(drop=True)
