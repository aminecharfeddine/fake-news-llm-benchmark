"""Baseline: TF-IDF (word 1-2 grams) + logistic regression."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from ..config import MODELS_DIR, RESULTS_DIR, SEED, ensure_dirs
from ..data import load_split
from ..evaluation import compute_metrics, save_predictions

MODEL_PATH = MODELS_DIR / "baseline.joblib"


def build_pipeline(C: float = 4.0) -> Pipeline:
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.9, max_features=200_000,
                                      sublinear_tf=True, strip_accents="unicode")),
            ("clf", LogisticRegression(C=C, max_iter=2000, random_state=SEED)),
        ]
    )


def tune_and_fit(train: pd.DataFrame, val: pd.DataFrame, grid=(0.5, 1.0, 4.0, 16.0)) -> tuple[Pipeline, dict]:
    """Pick C on the validation set, then refit on train + val."""
    scores = {}
    for C in grid:
        pipe = build_pipeline(C).fit(train["input"], train["label"])
        scores[C] = compute_metrics(val["label"], pipe.predict(val["input"]))["f1_macro"]
    best_C = max(scores, key=scores.get)
    final = build_pipeline(best_C).fit(pd.concat([train["input"], val["input"]]),
                                       pd.concat([train["label"], val["label"]]))
    return final, {"best_C": best_C, "val_f1_macro_by_C": scores}


def predict(model: Pipeline, texts) -> tuple[np.ndarray, np.ndarray, float]:
    """Return predicted labels, P(fake) and the average latency per article in ms."""
    texts = list(texts)
    start = time.perf_counter()
    prob = model.predict_proba(texts)[:, 1]
    latency_ms = (time.perf_counter() - start) * 1000 / max(len(texts), 1)
    return (prob >= 0.5).astype(int), prob, latency_ms


def top_features(model: Pipeline, k: int = 25) -> dict[str, list[tuple[str, float]]]:
    """Most influential n-grams for each class: a quick check for leakage and an explanation aid."""
    vocab = np.array(model.named_steps["tfidf"].get_feature_names_out())
    coef = model.named_steps["clf"].coef_[0]
    order = np.argsort(coef)
    return {
        "fake": [(vocab[i], round(float(coef[i]), 3)) for i in order[::-1][:k]],
        "real": [(vocab[i], round(float(coef[i]), 3)) for i in order[:k]],
    }


def explain(model: Pipeline, text: str, k: int = 8) -> list[tuple[str, float]]:
    """Per-article contributions (tf-idf weight x coefficient) of the strongest n-grams."""
    x = model.named_steps["tfidf"].transform([text])
    coef = model.named_steps["clf"].coef_[0]
    vocab = model.named_steps["tfidf"].get_feature_names_out()
    idx = x.nonzero()[1]
    contrib = sorted(((vocab[i], float(x[0, i] * coef[i])) for i in idx), key=lambda t: abs(t[1]), reverse=True)
    return contrib[:k]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate the TF-IDF + logistic regression baseline.")
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    args = parser.parse_args()

    ensure_dirs()
    train, val, test = load_split("train"), load_split("val"), load_split("test")
    model, tuning = tune_and_fit(train, val)
    joblib.dump(model, args.model_path)

    pred, prob, latency = predict(model, test["input"])
    save_predictions("tfidf_logreg", pd.DataFrame({
        "id": test["id"], "label": test["label"], "prob_fake": prob, "pred": pred,
        "latency_ms": latency, "cost_usd": 0.0,
    }))
    (RESULTS_DIR / "baseline_top_features.json").write_text(json.dumps(top_features(model), indent=2))

    m = compute_metrics(test["label"], pred, prob)
    print(f"Best C={tuning['best_C']}  test accuracy={m['accuracy']:.4f}  F1(fake)={m['f1_fake']:.4f}  "
          f"latency={latency:.3f} ms/article")


if __name__ == "__main__":
    main()
