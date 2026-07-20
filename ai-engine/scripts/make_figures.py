"""
Generate report-ready figures from the trained models' metrics JSON files.

Produces (into artifacts/figures/):
  - roc_lead_scoring.png, roc_churn.png     ROC curves
  - cm_<model>.png                          confusion-matrix heatmaps
  - importances_<model>.png                 top feature importances (lead, churn)
  - model_summary.png                       headline metric per model

Run (after training):
    python -m scripts.make_figures
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless — write PNGs, no display
import matplotlib.pyplot as plt
import numpy as np

ARTIFACTS = Path(__file__).resolve().parent.parent / "artifacts"
FIG_DIR = ARTIFACTS / "figures"

PRIMARY = "#4f46e5"   # indigo
ACCENT = "#0d9488"    # teal


def _load(name: str) -> dict | None:
    p = ARTIFACTS / name
    if not p.exists():
        print(f"  (skip) {name} not found — train the model first.")
        return None
    return json.loads(p.read_text())


def _save(fig, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path.relative_to(ARTIFACTS.parent)}")


def roc_figure(report: dict, title: str, fname: str) -> None:
    roc = report["metrics"].get("roc_curve")
    if not roc:
        return
    auc = report["metrics"]["roc_auc"]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(roc["fpr"], roc["tpr"], color=PRIMARY, lw=2.5, label=f"{report['model_name']} (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="#9ca3af", lw=1, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve — {title}")
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    _save(fig, fname)


def confusion_figure(report: dict, title: str, labels: list[str], fname: str) -> None:
    cm = np.array(report["metrics"]["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(1.4 * len(labels) + 2, 1.4 * len(labels) + 1.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {title}")
    thresh = cm.max() / 2 if cm.max() else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontweight="bold")
    _save(fig, fname)


def importance_figure(report: dict, title: str, fname: str, top: int = 10) -> None:
    feats = report.get("top_features")
    if not feats:
        return
    feats = feats[:top][::-1]
    names = [f["feature"] for f in feats]
    vals = [f["importance"] for f in feats]
    fig, ax = plt.subplots(figsize=(7, 0.45 * len(feats) + 1.5))
    ax.barh(names, vals, color=ACCENT)
    ax.set_xlabel("Importance")
    ax.set_title(f"Top Features — {title}")
    _save(fig, fname)


def summary_figure(rows: list[tuple[str, str, float]]) -> None:
    if not rows:
        return
    labels = [f"{n}\n({m})" for n, m, _ in rows]
    vals = [v for _, _, v in rows]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, vals, color=[PRIMARY, ACCENT, PRIMARY, ACCENT][:len(rows)])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("CBAS AI Engine — Headline Metric per Model")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}", ha="center", fontweight="bold")
    _save(fig, "model_summary.png")


def main() -> None:
    print("Generating figures...")
    summary: list[tuple[str, str, float]] = []

    lead = _load("lead_scoring_metrics.json")
    if lead:
        roc_figure(lead, "Lead Scoring", "roc_lead_scoring.png")
        confusion_figure(lead, "Lead Scoring", ["Not converted", "Converted"], "cm_lead_scoring.png")
        importance_figure(lead, "Lead Scoring", "importances_lead_scoring.png")
        summary.append(("Lead Scoring", "ROC-AUC", lead["metrics"]["roc_auc"]))

    churn = _load("churn_metrics.json")
    if churn:
        roc_figure(churn, "Churn", "roc_churn.png")
        confusion_figure(churn, "Churn", ["Stayed", "Churned"], "cm_churn.png")
        importance_figure(churn, "Churn", "importances_churn.png")
        summary.append(("Churn", "ROC-AUC", churn["metrics"]["roc_auc"]))

    churn_crm = _load("churn_crm_metrics.json")
    if churn_crm:
        roc_figure(churn_crm, "Churn (CRM-native)", "roc_churn_crm.png")
        confusion_figure(churn_crm, "Churn (CRM-native)", ["Active", "Lapsed"], "cm_churn_crm.png")
        importance_figure(churn_crm, "Churn (CRM-native)", "importances_churn_crm.png")

    sent = _load("sentiment_metrics.json")
    if sent:
        confusion_figure(sent, "Sentiment", sent["labels"], "cm_sentiment.png")
        summary.append(("Sentiment", "Accuracy", sent["metrics"]["accuracy"]))

    # Recommendations use Precision@K (a ranking metric on a different scale), so
    # it is intentionally omitted from this accuracy/AUC summary chart.

    summary_figure(summary)
    print(f"\nDone. Figures in {FIG_DIR}")


if __name__ == "__main__":
    main()
