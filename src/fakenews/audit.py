"""Leakage audit: how much of the "99% accuracy" on this dataset comes from source artifacts?

Trains the same TF-IDF baseline on the raw and on the cleaned data, and checks how well the
`subject` column alone predicts the label. Writes results/leakage_audit.md.
"""

from __future__ import annotations

import json

import pandas as pd

from .config import DATA_RAW, RESULTS_DIR, ensure_dirs
from .data import load_raw, prepare, split
from .evaluation import compute_metrics
from .models.baseline import build_pipeline, top_features


def subject_only_accuracy(raw: pd.DataFrame) -> float:
    """Predict each article's label as the majority label of its subject: a pure leak."""
    majority = raw.groupby("subject")["label"].agg(lambda s: s.mode().iloc[0])
    return float((raw["subject"].map(majority) == raw["label"]).mean())


def run_mode(raw: pd.DataFrame, mode: str) -> dict:
    df = prepare(raw, mode=mode)
    parts = split(df)
    model = build_pipeline().fit(parts["train"]["input"], parts["train"]["label"])
    test = parts["test"]
    m = compute_metrics(test["label"], model.predict(test["input"]))
    return {"n_articles": len(df), "metrics": m, "top_features": top_features(model, k=15)}


def main() -> None:
    ensure_dirs()
    raw = load_raw(DATA_RAW)
    results = {
        "subject_only_accuracy": subject_only_accuracy(raw),
        "raw": run_mode(raw, "raw"),
        "clean": run_mode(raw, "clean"),
    }
    (RESULTS_DIR / "leakage_audit.json").write_text(json.dumps(results, indent=2, default=float))

    r, c = results["raw"], results["clean"]
    fmt = lambda feats: ", ".join(f"`{w}`" for w, _ in feats[:10])  # noqa: E731
    md = f"""# Leakage audit

| Setting | Articles | Test accuracy | F1 (fake) |
|---|---:|---:|---:|
| `subject` column only (no text) | {len(raw):,} | {results['subject_only_accuracy']:.2%} | - |
| TF-IDF + LogReg, raw text | {r['n_articles']:,} | {r['metrics']['accuracy']:.2%} | {r['metrics']['f1_fake']:.4f} |
| TF-IDF + LogReg, cleaned text | {c['n_articles']:,} | {c['metrics']['accuracy']:.2%} | {c['metrics']['f1_fake']:.4f} |

**Strongest "real" n-grams, raw text:** {fmt(r['top_features']['real'])}

**Strongest "real" n-grams, cleaned text:** {fmt(c['top_features']['real'])}

**Strongest "fake" n-grams, raw text:** {fmt(r['top_features']['fake'])}

**Strongest "fake" n-grams, cleaned text:** {fmt(c['top_features']['fake'])}
"""
    (RESULTS_DIR / "leakage_audit.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
