"""Aggregate every model's predictions into comparison tables and figures.

Two tables are produced:
- full test set, for the models evaluated on all test articles (baseline, transformer);
- head-to-head on the LLM subsample, where every model is scored on exactly the same articles.
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .config import RESULTS_DIR, ensure_dirs  # noqa: E402
from .data import load_split  # noqa: E402
from .evaluation import compute_metrics, load_predictions  # noqa: E402

PRETTY = {"tfidf_logreg": "TF-IDF + LogReg", "distilbert_finetuned": "DistilBERT (fine-tuned)"}


def pretty(name: str) -> str:
    if name in PRETTY:
        return PRETTY[name]
    if name.startswith("llm_"):
        model, shots = name[4:].rsplit("_", 1)
        return f"{model} ({shots.replace('shot', '-shot')})"
    return name


def summarize(preds: pd.DataFrame) -> dict:
    m = compute_metrics(preds["label"], preds["pred"], preds["prob_fake"])
    m["latency_ms_per_article"] = float(preds["latency_ms"].mean())
    m["cost_per_1k_usd"] = float(preds["cost_usd"].mean() * 1000)
    return m


def table(rows: dict[str, dict]) -> str:
    lines = ["| Model | n | Accuracy | F1 (fake) | Recall (fake) | FPR | Latency (ms/article) | Cost / 1k articles |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, m in sorted(rows.items(), key=lambda kv: -kv[1]["f1_fake"]):
        lines.append(
            f"| {pretty(name)} | {m['n']:,} | {m['accuracy']:.2%} | {m['f1_fake']:.4f} | {m['recall_fake']:.2%} | "
            f"{m['false_positive_rate']:.2%} | {m['latency_ms_per_article']:.2f} | ${m['cost_per_1k_usd']:.2f} |"
        )
    return "\n".join(lines)


def plot_confusions(preds: dict[str, pd.DataFrame], path) -> None:
    fig, axes = plt.subplots(1, len(preds), figsize=(3.6 * len(preds), 3.4), squeeze=False)
    for ax, (name, p) in zip(axes[0], preds.items()):
        m = compute_metrics(p["label"], p["pred"])
        cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
        ax.imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, f"{v:,}", ha="center", va="center", color="white" if v > cm.max() / 2 else "black")
        ax.set_xticks([0, 1], ["real", "fake"])
        ax.set_yticks([0, 1], ["real", "fake"])
        ax.set_xlabel("predicted")
        ax.set_ylabel("actual")
        ax.set_title(pretty(name), fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_cost_vs_f1(rows: dict[str, dict], path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, m in rows.items():
        cost = max(m["cost_per_1k_usd"], 0.001)  # free local models sit at the left edge of the log axis
        ax.scatter(cost, m["f1_fake"], s=60)
        ax.annotate(pretty(name), (cost, m["f1_fake"]), textcoords="offset points", xytext=(6, 4), fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("API cost per 1,000 articles (USD, log scale; local models shown at $0.001)")
    ax.set_ylabel("F1 (fake)")
    ax.set_title("Quality vs cost, same test articles")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    ensure_dirs()
    preds = load_predictions()
    if not preds:
        raise SystemExit("No predictions found in results/predictions. Train at least one model first.")
    n_test = len(load_split("test"))

    full = {k: summarize(p) for k, p in preds.items() if len(p) == n_test}
    llm_ids = [set(p["id"]) for k, p in preds.items() if k.startswith("llm_")]
    common = set.intersection(*llm_ids) if llm_ids else set()
    h2h_preds = {k: p[p["id"].isin(common)].sort_values("id") for k, p in preds.items()} if common else {}
    h2h = {k: summarize(p) for k, p in h2h_preds.items() if len(p) == len(common)}

    parts = ["# Results\n"]
    if full:
        parts += [f"## Full test set ({n_test:,} articles)\n", table(full), ""]
    if h2h:
        parts += [f"## Head-to-head on the LLM subsample ({len(common):,} articles, same rows for every model)\n",
                  table(h2h), "", "![Quality vs cost](cost_vs_f1.png)", ""]
        plot_cost_vs_f1(h2h, RESULTS_DIR / "cost_vs_f1.png")
    plot_confusions({k: p for k, p in (h2h_preds or preds).items() if len(p)}, RESULTS_DIR / "confusion_matrices.png")
    parts += ["![Confusion matrices](confusion_matrices.png)", ""]

    (RESULTS_DIR / "results.md").write_text("\n".join(parts))
    (RESULTS_DIR / "metrics.json").write_text(json.dumps({"full_test": full, "head_to_head": h2h}, indent=2))
    print("\n".join(parts))


if __name__ == "__main__":
    main()
