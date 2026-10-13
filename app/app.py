"""Gradio demo: paste an article and compare the three classifiers side by side.

Run locally with `python app/app.py` after training the models. Models that are not
available are skipped: the LLM is used only when OPENAI_API_KEY and LLM_MODEL are set.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import gradio as gr
import joblib

from fakenews.config import MODELS_DIR
from fakenews.data import clean_text, clean_title
from fakenews.models import baseline

MODELS: dict[str, object] = {}

if (MODELS_DIR / "baseline.joblib").exists():
    MODELS["TF-IDF + LogReg"] = joblib.load(MODELS_DIR / "baseline.joblib")

if (MODELS_DIR / "transformer" / "config.json").exists():
    try:
        from fakenews.models import transformer
        MODELS["DistilBERT (fine-tuned)"] = transformer.load(MODELS_DIR / "transformer")
    except ImportError:
        pass

if os.environ.get("OPENAI_API_KEY") and os.environ.get("LLM_MODEL"):
    from fakenews.models.llm import LLMClassifier
    MODELS[f"{os.environ['LLM_MODEL']} (zero-shot)"] = LLMClassifier(os.environ["LLM_MODEL"])


def classify(title: str, text: str):
    article = f"{clean_title(title)}. {clean_text(text)}".strip(". ")
    if len(article) < 50:
        return "Paste a headline and at least a few sentences of the article.", ""
    rows = ["| Model | Verdict | P(fake) | Latency |", "|---|---|---:|---:|"]
    for name, model in MODELS.items():
        start = time.perf_counter()
        if name.startswith("TF-IDF"):
            prob = float(model.predict_proba([article])[0, 1])
        elif name.startswith("DistilBERT"):
            prob = float(transformer.predict_proba(*model, [article])[0][0])
        else:
            out = model.classify(article)
            prob = out["prob_fake"]
        ms = (time.perf_counter() - start) * 1000
        verdict = "🟥 fake" if prob >= 0.5 else "🟩 real"
        rows.append(f"| {name} | {verdict} | {prob:.2f} | {ms:.0f} ms |")

    explanation = ""
    if "TF-IDF + LogReg" in MODELS:
        contrib = baseline.explain(MODELS["TF-IDF + LogReg"], article)
        explanation = "**Words that weighed most in the TF-IDF model's decision**\n\n" + "\n".join(
            f"- `{w}` → {'fake' if c > 0 else 'real'} ({c:+.3f})" for w, c in contrib)
    return "\n".join(rows), explanation


with gr.Blocks(title="Fake News Detection Benchmark") as demo:
    gr.Markdown(
        "# Fake news detection: classical ML vs fine-tuned transformer vs LLM\n"
        "Paste a news headline and article. Source markers (agency datelines, image credits, URLs) are stripped "
        "before classification, as in training. Models were trained on 2016-2017 US political news, so "
        "predictions on other topics or periods are less reliable."
    )
    with gr.Row():
        with gr.Column():
            title = gr.Textbox(label="Headline")
            text = gr.Textbox(label="Article text", lines=12)
            btn = gr.Button("Classify", variant="primary")
        with gr.Column():
            table = gr.Markdown()
            expl = gr.Markdown()
    btn.click(classify, [title, text], [table, expl])

if __name__ == "__main__":
    if not MODELS:
        raise SystemExit("No trained model found in models/. Run `make data baseline` first.")
    demo.launch()
