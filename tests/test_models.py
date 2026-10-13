import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from fakenews import data
from fakenews.evaluation import compute_metrics
from fakenews.models import baseline
from fakenews.models.llm import LLMClassifier, build_messages, few_shot_examples, parse_response


def test_compute_metrics_basic():
    m = compute_metrics([0, 0, 1, 1], [0, 1, 1, 1], [0.1, 0.6, 0.9, 0.8])
    assert m["accuracy"] == 0.75
    assert m["recall_fake"] == 1.0
    assert m["false_positive_rate"] == 0.5
    assert m["roc_auc"] == 1.0
    assert (m["tp"], m["fp"], m["tn"], m["fn"]) == (2, 1, 1, 0)


def test_baseline_learns_and_explains(processed_dir):
    train, val, test = (data.load_split(n) for n in ("train", "val", "test"))
    model, tuning = baseline.tune_and_fit(train, val, grid=(1.0, 4.0))
    pred, prob, latency = baseline.predict(model, test["input"])
    assert compute_metrics(test["label"], pred)["accuracy"] > 0.9
    assert ((prob >= 0) & (prob <= 1)).all() and latency >= 0
    feats = baseline.top_features(model, k=5)
    assert len(feats["fake"]) == 5 and len(feats["real"]) == 5
    assert all("reuters" not in w for w, _ in feats["real"])  # the leak is gone
    assert baseline.explain(model, test["input"].iloc[0])


@pytest.mark.parametrize("content, label, prob", [
    ('{"label": "fake", "confidence": 0.8}', 1, 0.8),
    ('{"label": "REAL", "confidence": 0.9}', 0, pytest.approx(0.1)),
    ('{"label": "fake", "confidence": 7}', 1, 1.0),
    ("I think this is fake.", 1, 0.5),
    ("no idea", None, None),
])
def test_parse_response(content, label, prob):
    got_label, got_prob = parse_response(content)
    assert got_label == label
    if prob is None:
        assert np.isnan(got_prob)
    else:
        assert got_prob == prob


def test_few_shot_examples_are_balanced(processed_dir):
    train = data.load_split("train")
    train = train.assign(input=train["input"].str.pad(700))  # synthetic articles are short
    ex = few_shot_examples(train, 4)
    assert [lbl for _, lbl in ex] == ["fake", "real", "fake", "real"]
    msgs = build_messages("article", ex)
    assert msgs[0]["role"] == "system" and msgs[-1]["content"].endswith("article") and len(msgs) == 10


class FakeClient:
    """Mimics openai.OpenAI().chat.completions.create."""

    def __init__(self):
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        text = kwargs["messages"][-1]["content"].lower()
        label = "fake" if "shocking" in text or "exposed" in text else "real"
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps({"label": label, "confidence": 0.9})))],
            usage=SimpleNamespace(prompt_tokens=1000, completion_tokens=10),
        )


def test_llm_classifier_cost_and_cache(processed_dir, tmp_path):
    from fakenews.models.llm import run

    test = data.load_split("test")
    client = FakeClient()
    clf = LLMClassifier("test-model", client=client, price_in=0.5, price_out=2.0)
    cache = tmp_path / "cache.jsonl"
    out = run(clf, test, cache, workers=2)
    assert len(out) == len(test) and client.calls == len(test)
    assert out["cost_usd"].iloc[0] == pytest.approx((1000 * 0.5 + 10 * 2.0) / 1e6)
    run(clf, test, cache, workers=2)  # second run is served from the cache
    assert client.calls == len(test)


def test_transformer_pipeline_end_to_end(processed_dir, tmp_path):
    """Trains a tiny randomly initialised BERT (no download) to check the training/prediction code."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers

    from fakenews.models import transformer

    train, val, test = (data.load_split(n) for n in ("train", "val", "test"))
    tok = Tokenizer(models.WordLevel(unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.train_from_iterator(train["input"], trainers.WordLevelTrainer(
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]))
    hf_tok = transformers.PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]", pad_token="[PAD]",
                                                  cls_token="[CLS]", sep_token="[SEP]", mask_token="[MASK]")
    cfg = transformers.BertConfig(vocab_size=tok.get_vocab_size(), hidden_size=32, num_hidden_layers=1,
                                  num_attention_heads=2, intermediate_size=64, max_position_embeddings=128)
    base = tmp_path / "tiny-bert"
    transformers.BertForSequenceClassification(cfg).save_pretrained(base)
    hf_tok.save_pretrained(base)

    out = tmp_path / "ft"
    info = transformer.train(train, val, model_name=str(base), out_dir=out, epochs=2, lr=1e-3, batch_size=16, max_len=64)
    assert len(info["history"]) == 2 and (out / "training_info.json").exists()
    model, tokenizer = transformer.load(out)
    prob, latency = transformer.predict_proba(model, tokenizer, test["input"], max_len=64)
    assert prob.shape == (len(test),) and latency > 0
    assert compute_metrics(test["label"], (prob >= 0.5).astype(int))["accuracy"] > 0.8
