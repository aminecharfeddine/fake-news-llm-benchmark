"""Leakage audit: how much of the "99% accuracy" on this dataset comes from source artifacts?

Trains the same TF-IDF baseline on the raw and on the cleaned data, and checks how well the
`subject` column alone predicts the label. Writes results/leakage_audit.md.
"""

from __future__ import annotations

import json
import re

import pandas as pd

from .config import DATA_RAW, RESULTS_DIR, ensure_dirs
from .data import load_raw, prepare, split
from .evaluation import compute_metrics
from .models.baseline import build_pipeline, top_features


def subject_only_accuracy(raw: pd.DataFrame) -> float:
    """Predict each article's label as the majority label of its subject: a pure leak."""
    majority = raw.groupby("subject")["label"].agg(lambda s: s.mode().iloc[0])
    return float((raw["subject"].map(majority) == raw["label"]).mean())


# Wire-service house style: "said on Wednesday", "President Donald Trump". Not a source marker
# strictly speaking, but a convention of one outlet that a model can latch onto.
_WIRE_STYLE = re.compile(r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", re.IGNORECASE)


def mask_wire_style(text: str) -> str:
    return _WIRE_STYLE.sub(" ", text)


def run_mode(raw: pd.DataFrame, mode: str, mask_style: bool = False, strategy: str = "random") -> dict:
    df = prepare(raw, mode=mode)
    if mask_style:
        df["input"] = df["input"].map(mask_wire_style)
    parts = split(df, strategy=strategy)
    model = build_pipeline().fit(parts["train"]["input"], parts["train"]["label"])
    test = parts["test"]
    m = compute_metrics(test["label"], model.predict(test["input"]))
    return {"n_articles": len(df), "metrics": m, "top_features": top_features(model, k=15),
            "test_period": [str(test["date"].min().date()), str(test["date"].max().date())]}


def main() -> None:
    ensure_dirs()
    raw = load_raw(DATA_RAW)
    results = {
        "subject_only_accuracy": subject_only_accuracy(raw),
        "raw": run_mode(raw, "raw"),
        "clean": run_mode(raw, "clean"),
        "clean_style_masked": run_mode(raw, "clean", mask_style=True),
        "clean_temporal": run_mode(raw, "clean", strategy="temporal"),
    }
    (RESULTS_DIR / "leakage_audit.json").write_text(json.dumps(results, indent=2, default=float))

    r, c, s, t = results["raw"], results["clean"], results["clean_style_masked"], results["clean_temporal"]
    fmt = lambda feats: ", ".join(f"`{w}`" for w, _ in feats[:10])  # noqa: E731
    md = f"""# Leakage audit

| Setting | Articles | Test accuracy | F1 (fake) | FPR |
|---|---:|---:|---:|---:|
| `subject` column only (no text) | {len(raw):,} | {results['subject_only_accuracy']:.2%} | - | - |
| TF-IDF + LogReg, raw text | {r['n_articles']:,} | {r['metrics']['accuracy']:.2%} | {r['metrics']['f1_fake']:.4f} | {r['metrics']['false_positive_rate']:.2%} |
| TF-IDF + LogReg, cleaned text | {c['n_articles']:,} | {c['metrics']['accuracy']:.2%} | {c['metrics']['f1_fake']:.4f} | {c['metrics']['false_positive_rate']:.2%} |
| TF-IDF + LogReg, cleaned text + weekday names masked | {s['n_articles']:,} | {s['metrics']['accuracy']:.2%} | {s['metrics']['f1_fake']:.4f} | {s['metrics']['false_positive_rate']:.2%} |
| TF-IDF + LogReg, cleaned text, temporal split (test {t['test_period'][0]} to {t['test_period'][1]}) | {t['n_articles']:,} | {t['metrics']['accuracy']:.2%} | {t['metrics']['f1_fake']:.4f} | {t['metrics']['false_positive_rate']:.2%} |

Random splits are stratified (44.6% fake in every split). The temporal split trains on the oldest 70% of articles and tests on the most recent 15%, where only {t['metrics']['tp'] + t['metrics']['fn']:,} of {t['metrics']['n']:,} test articles are fake: the class balance shifts over time, which is part of what makes it harder.

**Strongest "real" n-grams, raw text:** {fmt(r['top_features']['real'])}

**Strongest "real" n-grams, cleaned text:** {fmt(c['top_features']['real'])}

**Strongest "fake" n-grams, raw text:** {fmt(r['top_features']['fake'])}

**Strongest "fake" n-grams, cleaned text:** {fmt(c['top_features']['fake'])}
"""
    (RESULTS_DIR / "leakage_audit.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
