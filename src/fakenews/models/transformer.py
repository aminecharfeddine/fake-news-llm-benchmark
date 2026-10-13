"""Fine-tuned transformer classifier (DistilBERT by default), with a plain PyTorch training loop.

Training on CPU is slow; on a free Colab/Kaggle GPU one epoch over the full training set takes
roughly 10-15 minutes. Use --max-train to train on a subset when iterating.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from ..config import MODELS_DIR, SEED, ensure_dirs
from ..data import load_split
from ..evaluation import compute_metrics, save_predictions

DEFAULT_MODEL = "distilbert-base-uncased"
DEFAULT_OUT = MODELS_DIR / "transformer"


def device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels=None):
        self.texts = list(texts)
        self.labels = None if labels is None else list(labels)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        return (self.texts[i], -1 if self.labels is None else self.labels[i])


def _collate(tokenizer, max_len):
    def collate(batch):
        texts, labels = zip(*batch)
        enc = tokenizer(list(texts), truncation=True, max_length=max_len, padding=True, return_tensors="pt")
        enc["labels"] = torch.tensor(labels)
        return enc
    return collate


@torch.no_grad()
def predict_proba(model, tokenizer, texts, max_len: int = 256, batch_size: int = 32) -> tuple[np.ndarray, float]:
    """Return P(fake) per text and the average latency per article in ms."""
    dev = next(model.parameters()).device
    model.eval()
    loader = DataLoader(TextDataset(texts), batch_size=batch_size, collate_fn=_collate(tokenizer, max_len))
    probs, start = [], time.perf_counter()
    for batch in loader:
        batch.pop("labels")
        logits = model(**{k: v.to(dev) for k, v in batch.items()}).logits
        probs.append(torch.softmax(logits, dim=-1)[:, 1].cpu().numpy())
    latency_ms = (time.perf_counter() - start) * 1000 / max(len(texts), 1)
    return (np.concatenate(probs) if probs else np.array([])), latency_ms


def train(train_df: pd.DataFrame, val_df: pd.DataFrame, model_name: str = DEFAULT_MODEL, out_dir: Path = DEFAULT_OUT,
          epochs: int = 2, lr: float = 3e-5, batch_size: int = 16, max_len: int = 256) -> dict:
    """Fine-tune, keep the checkpoint with the best validation macro-F1, save it to out_dir."""
    torch.manual_seed(SEED)
    dev = device()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2, id2label={0: "real", 1: "fake"}, label2id={"real": 0, "fake": 1}
    ).to(dev)

    loader = DataLoader(TextDataset(train_df["input"], train_df["label"]), batch_size=batch_size, shuffle=True,
                        collate_fn=_collate(tokenizer, max_len), generator=torch.Generator().manual_seed(SEED))
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = epochs * len(loader)
    sched = get_linear_schedule_with_warmup(optim, int(0.06 * steps), steps)

    history, best = [], -1.0
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for batch in loader:
            out = model(**{k: v.to(dev) for k, v in batch.items()})
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step(), sched.step(), optim.zero_grad()
            total += out.loss.item()
        prob, _ = predict_proba(model, tokenizer, val_df["input"], max_len=max_len)
        val = compute_metrics(val_df["label"], (prob >= 0.5).astype(int), prob)
        history.append({"epoch": epoch, "train_loss": total / len(loader), "val_f1_macro": val["f1_macro"]})
        print(f"epoch {epoch}: train loss {total / len(loader):.4f}  val F1 macro {val['f1_macro']:.4f}")
        if val["f1_macro"] > best:
            best = val["f1_macro"]
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            model.save_pretrained(out_dir)
            tokenizer.save_pretrained(out_dir)
    info = {"base_model": model_name, "max_len": max_len, "epochs": epochs, "lr": lr, "history": history}
    (Path(out_dir) / "training_info.json").write_text(json.dumps(info, indent=2))
    return info


def load(model_dir: Path = DEFAULT_OUT):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device())
    return model, tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune and evaluate a transformer classifier.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face model name or local path")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--max-train", type=int, default=None, help="Subsample the training set (faster runs)")
    args = parser.parse_args()

    ensure_dirs()
    train_df, val_df, test = load_split("train"), load_split("val"), load_split("test")
    if args.max_train and args.max_train < len(train_df):
        train_df = train_df.groupby("label", group_keys=False).apply(
            lambda g: g.sample(n=min(len(g), args.max_train // 2), random_state=SEED))
    train(train_df, val_df, args.model, args.out_dir, args.epochs, args.lr, args.batch_size, args.max_len)

    model, tokenizer = load(args.out_dir)
    prob, latency = predict_proba(model, tokenizer, test["input"], max_len=args.max_len)
    pred = (prob >= 0.5).astype(int)
    save_predictions("distilbert_finetuned", pd.DataFrame({
        "id": test["id"], "label": test["label"], "prob_fake": prob, "pred": pred,
        "latency_ms": latency, "cost_usd": 0.0,
    }))
    m = compute_metrics(test["label"], pred, prob)
    print(f"test accuracy={m['accuracy']:.4f}  F1(fake)={m['f1_fake']:.4f}  latency={latency:.2f} ms/article "
          f"on {device().type}")


if __name__ == "__main__":
    main()
