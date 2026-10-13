"""LLM classifier through the OpenAI API, zero-shot or few-shot, with cost and latency tracking.

Calls are cached in results/llm_cache/ so re-running the benchmark never pays twice for the
same article. Token prices are passed on the command line because they change over time:
check the provider's pricing page before running.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import RESULTS_DIR, SEED, ensure_dirs
from ..data import load_split
from ..evaluation import compute_metrics, save_predictions

CACHE_DIR = RESULTS_DIR / "llm_cache"

SYSTEM_PROMPT = (
    "You are a fact-checking assistant. Classify a news article as 'fake' (fabricated, misleading or "
    "hyper-partisan content presented as news) or 'real' (professional, factual news reporting). "
    "Judge the content and writing only; the source and publication details have been removed. "
    'Answer with JSON only: {"label": "fake" | "real", "confidence": <number between 0 and 1>}.'
)


def build_messages(article: str, examples: list[tuple[str, str]] | None = None, max_chars: int = 2500) -> list[dict]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for text, label in examples or []:
        messages.append({"role": "user", "content": f"Article:\n{text[:max_chars // 2]}"})
        messages.append({"role": "assistant", "content": json.dumps({"label": label, "confidence": 0.9})})
    messages.append({"role": "user", "content": f"Article:\n{article[:max_chars]}"})
    return messages


def parse_response(content: str) -> tuple[int | None, float]:
    """Return (label, P(fake)). Label is None when the answer cannot be parsed."""
    try:
        data = json.loads(content)
        label = str(data.get("label", "")).strip().lower()
        conf = float(data.get("confidence", 0.5))
    except (json.JSONDecodeError, TypeError, ValueError, AttributeError):
        match = re.search(r"\b(fake|real)\b", content or "", re.IGNORECASE)
        label, conf = (match.group(1).lower(), 0.5) if match else ("", 0.5)
    conf = min(max(conf, 0.0), 1.0)
    if label == "fake":
        return 1, conf
    if label == "real":
        return 0, 1.0 - conf
    return None, float("nan")


def few_shot_examples(train: pd.DataFrame, k: int) -> list[tuple[str, str]]:
    """k balanced, short training examples (alternating labels)."""
    if k <= 0:
        return []
    pool = train[train["input"].str.len().between(600, 1500)]
    fake = pool[pool["label"] == 1].sample(n=(k + 1) // 2, random_state=SEED)
    real = pool[pool["label"] == 0].sample(n=k // 2, random_state=SEED)
    rows = [r for pair in zip(fake.itertuples(), real.itertuples()) for r in pair]
    rows += list(fake.itertuples())[len(real):]
    return [(r.input, "fake" if r.label == 1 else "real") for r in rows[:k]]


class LLMClassifier:
    def __init__(self, model: str, shots: int = 0, train: pd.DataFrame | None = None,
                 price_in: float = 0.0, price_out: float = 0.0, client=None, max_chars: int = 2500):
        """price_in / price_out: USD per 1M input / output tokens."""
        if client is None:
            from openai import OpenAI
            client = OpenAI()  # reads OPENAI_API_KEY
        self.client, self.model, self.max_chars = client, model, max_chars
        self.examples = few_shot_examples(train, shots) if shots and train is not None else []
        self.price_in, self.price_out = price_in, price_out

    def classify(self, article: str, retries: int = 4) -> dict:
        messages = build_messages(article, self.examples, self.max_chars)
        for attempt in range(retries):
            try:
                start = time.perf_counter()
                resp = self.client.chat.completions.create(
                    model=self.model, messages=messages, temperature=0, max_tokens=30,
                    response_format={"type": "json_object"},
                )
                latency = (time.perf_counter() - start) * 1000
                break
            except Exception:  # rate limits / transient network errors
                if attempt == retries - 1:
                    raise
                time.sleep(2 ** attempt)
        label, prob = parse_response(resp.choices[0].message.content)
        usage = resp.usage
        cost = (usage.prompt_tokens * self.price_in + usage.completion_tokens * self.price_out) / 1e6
        return {"pred": label, "prob_fake": prob, "latency_ms": latency, "cost_usd": cost,
                "prompt_tokens": usage.prompt_tokens, "completion_tokens": usage.completion_tokens}


def run(clf: LLMClassifier, df: pd.DataFrame, cache_path: Path, workers: int = 8) -> pd.DataFrame:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cached = {}
    if cache_path.exists():
        for line in cache_path.read_text().splitlines():
            rec = json.loads(line)
            cached[rec["id"]] = rec
    todo = df[~df["id"].isin(cached)]

    def job(row):
        return {"id": int(row.id), **clf.classify(row.input)}

    with ThreadPoolExecutor(max_workers=workers) as pool, cache_path.open("a") as f:
        for i, rec in enumerate(pool.map(job, todo.itertuples()), 1):
            f.write(json.dumps(rec) + "\n")
            f.flush()
            cached[rec["id"]] = rec
            if i % 50 == 0:
                print(f"  {i}/{len(todo)} articles classified")
    out = df[["id", "label"]].merge(pd.DataFrame([cached[i] for i in df["id"]]), on="id")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an LLM (OpenAI API) on a test subsample.")
    parser.add_argument("--model", required=True, help="OpenAI model name, e.g. the current small/cheap model")
    parser.add_argument("--shots", type=int, default=0, help="Number of few-shot examples (0 = zero-shot)")
    parser.add_argument("--n", type=int, default=500, help="Number of test articles (stratified sample)")
    parser.add_argument("--price-in", type=float, required=True, help="USD per 1M input tokens")
    parser.add_argument("--price-out", type=float, required=True, help="USD per 1M output tokens")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY first.")
    ensure_dirs()
    train, test = load_split("train"), load_split("test")
    sample = test.groupby("label", group_keys=False).apply(
        lambda g: g.sample(n=min(len(g), args.n // 2), random_state=SEED))

    clf = LLMClassifier(args.model, args.shots, train, args.price_in, args.price_out)
    name = f"llm_{args.model.replace('/', '-')}_{args.shots}shot"
    out = run(clf, sample, CACHE_DIR / f"{name}.jsonl", args.workers)

    failed = out["pred"].isna().sum()
    out["pred"] = out["pred"].fillna(1 - out["label"]).astype(int)  # unparseable answers count as errors
    save_predictions(name, out)
    m = compute_metrics(out["label"], out["pred"], out["prob_fake"])
    print(f"{name}: accuracy={m['accuracy']:.4f}  F1(fake)={m['f1_fake']:.4f}  unparsed={failed}  "
          f"cost per 1k articles=${out['cost_usd'].mean() * 1000:.2f}  median latency={np.median(out['latency_ms']):.0f} ms")


if __name__ == "__main__":
    main()
