"""Metrics and prediction files shared by every model."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .config import PREDICTIONS_DIR

PRED_COLUMNS = ["id", "label", "prob_fake", "pred", "latency_ms", "cost_usd"]


def compute_metrics(y_true, y_pred, prob_fake=None) -> dict[str, float]:
    """Classification metrics with "fake" (label 1) as the positive class."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics = {
        "n": int(len(y_true)),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_fake": precision_score(y_true, y_pred, zero_division=0),
        "recall_fake": recall_score(y_true, y_pred, zero_division=0),
        "f1_fake": f1_score(y_true, y_pred, zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "false_positive_rate": fp / (fp + tn) if (fp + tn) else 0.0,
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }
    if prob_fake is not None:
        prob = np.asarray(prob_fake, dtype=float)
        valid = ~np.isnan(prob)
        if valid.any() and len(np.unique(y_true[valid])) == 2:
            metrics["roc_auc"] = roc_auc_score(y_true[valid], prob[valid])
    return metrics


def save_predictions(name: str, df: pd.DataFrame, out_dir: Path = PREDICTIONS_DIR) -> Path:
    """Write one model's test-set predictions so the report can compare models on identical rows."""
    missing = set(PRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Predictions are missing columns: {sorted(missing)}")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.csv"
    df[PRED_COLUMNS].to_csv(path, index=False)
    return path


def load_predictions(out_dir: Path = PREDICTIONS_DIR) -> dict[str, pd.DataFrame]:
    return {p.stem: pd.read_csv(p) for p in sorted(Path(out_dir).glob("*.csv"))}
