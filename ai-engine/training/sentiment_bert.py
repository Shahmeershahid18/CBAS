"""
Sentiment Analysis — DistilBERT fine-tune (the plan's optional accuracy upgrade).

Fine-tunes a pretrained DistilBERT transformer on the review data instead of the
TF-IDF + Logistic Regression baseline. No language model is trained from scratch;
a pretrained model is adapted (transfer learning). Trains on a balanced subset so
it is tractable on CPU. Saves the model + a metrics report; the lightweight
TF-IDF model remains the default served endpoint.

Run:  python -m training.sentiment_bert            # 3000/class, 2 epochs
      python -m training.sentiment_bert --per-class 2000 --epochs 2
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    Trainer,
    TrainingArguments,
)

from app.config import ARTIFACTS_DIR
from training.sentiment import DEFAULT_CSV, clean_text, load_dataset

LABELS = ["negative", "neutral", "positive"]
L2I = {l: i for i, l in enumerate(LABELS)}
MODEL_NAME = "distilbert-base-uncased"
OUT_DIR = ARTIFACTS_DIR / "sentiment_bert"


class ReviewDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
    }


def main(a) -> None:
    df, source = load_dataset(DEFAULT_CSV, force_synthetic=False)
    df["text"] = df["text"].map(clean_text)
    df = df[df["text"].str.len() > 0]

    if a.natural:
        # NATURAL distribution: sample up to max_rows keeping class proportions, then
        # a stratified split. The test set matches the TF-IDF baseline's, so accuracy
        # is directly comparable (both positive-heavy).
        data = df.sample(min(a.max_rows, len(df)), random_state=42) if len(df) > a.max_rows else df
        mode = f"natural dist, {len(data):,} rows"
    else:
        # BALANCED subset (each class equally represented) — fairer for macro-F1.
        parts = [df[df["sentiment"] == lab].sample(min(a.per_class, (df["sentiment"] == lab).sum()), random_state=42)
                 for lab in LABELS]
        data = pd.concat(parts).sample(frac=1, random_state=42)
        mode = f"balanced, {a.per_class}/class"

    y = data["sentiment"].map(L2I).to_numpy()
    gpu = torch.cuda.is_available()
    print(f"Fine-tuning DistilBERT on {len(data):,} reviews from {source} "
          f"({mode}, {a.epochs} epochs, batch {a.batch}, "
          f"{'GPU: ' + torch.cuda.get_device_name(0) if gpu else 'CPU'})")

    X_tr, X_te, y_tr, y_te = train_test_split(
        data["text"].tolist(), y, test_size=0.2, random_state=42, stratify=y)

    tok = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
    enc_tr = tok(X_tr, truncation=True, padding=True, max_length=a.max_len)
    enc_te = tok(X_te, truncation=True, padding=True, max_length=a.max_len)
    ds_tr = ReviewDataset(enc_tr, y_tr.tolist())
    ds_te = ReviewDataset(enc_te, y_te.tolist())

    model = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)

    args = TrainingArguments(
        output_dir=str(ARTIFACTS_DIR / "_bert_ckpt"),
        num_train_epochs=a.epochs,
        per_device_train_batch_size=a.batch,
        per_device_eval_batch_size=a.batch * 2,
        gradient_accumulation_steps=a.grad_accum,
        learning_rate=5e-5,
        weight_decay=0.01,
        logging_steps=100,
        report_to=[],
        save_strategy="no",
        fp16=a.fp16 and gpu,          # halves VRAM use; helps fit the 4 GB P1000
    )
    trainer = Trainer(model=model, args=args, train_dataset=ds_tr,
                      eval_dataset=ds_te, compute_metrics=compute_metrics)
    trainer.train()

    # Final held-out evaluation.
    logits = trainer.predict(ds_te).predictions
    preds = np.argmax(logits, axis=1)
    acc = float(accuracy_score(y_te, preds))
    macro = float(f1_score(y_te, preds, average="macro"))
    print(f"\nDistilBERT -> accuracy {acc:.4f}  macro-F1 {macro:.4f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(OUT_DIR)
    tok.save_pretrained(OUT_DIR)
    report = {
        "model_name": "distilbert_finetuned",
        "data_source": source,
        "labels": LABELS,
        "metrics": {
            "accuracy": round(acc, 4),
            "macro_f1": round(macro, 4),
            "per_class": classification_report(y_te, preds, target_names=LABELS, output_dict=True, zero_division=0),
            "confusion_matrix": confusion_matrix(y_te, preds).tolist(),
        },
        "trained_rows": int(len(X_tr)),
        "eval_distribution": "natural" if a.natural else "balanced",
        "note": "Transfer-learning fine-tune of pretrained DistilBERT.",
    }
    (ARTIFACTS_DIR / "sentiment_bert_metrics.json").write_text(json.dumps(report, indent=2))
    print(f"Saved model -> {OUT_DIR}")
    print(f"Metrics     -> {ARTIFACTS_DIR / 'sentiment_bert_metrics.json'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Fine-tune DistilBERT for sentiment.")
    p.add_argument("--natural", action="store_true",
                   help="Train/eval on the natural class distribution (accuracy comparable to the TF-IDF baseline).")
    p.add_argument("--max-rows", type=int, default=60000, help="Max rows in --natural mode.")
    p.add_argument("--per-class", type=int, default=3000, help="Rows per class in balanced mode.")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--max-len", type=int, default=128)
    p.add_argument("--batch", type=int, default=8, help="Per-device batch size (8 fits a 4 GB GPU).")
    p.add_argument("--grad-accum", type=int, default=2, help="Gradient accumulation steps.")
    p.add_argument("--fp16", action="store_true", help="Mixed precision (saves VRAM on GPU).")
    main(p.parse_args())
