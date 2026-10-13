"""Synthetic stand-in for the ISOT dataset that reproduces its leakage artifacts."""

import random

import pandas as pd
import pytest

REAL_WORDS = "senate committee approved budget officials said statement minister talks agreement spokesman".split()
FAKE_WORDS = "shocking truth exposed hillary liberals destroy outrage secret elite media cover".split()
NEUTRAL = "the a of to in and that for on with was is as by it this government president".split()


def _article(words, rng, n=80):
    return " ".join(rng.choice(words + NEUTRAL * 2) for _ in range(n)).capitalize() + "."


def make_raw(n_per_class=120, seed=0):
    rng = random.Random(seed)
    real, fake = [], []
    for i in range(n_per_class):
        real.append({
            "title": f"Officials reach agreement on plan {i}",
            "text": f"WASHINGTON (Reuters) - {_article(REAL_WORDS, rng)}",
            "subject": "politicsNews",
            "date": f"December {i % 28 + 1}, 2017 ",
        })
        fake.append({
            "title": f"SHOCKING: Secret plan {i} EXPOSED (VIDEO)",
            "text": f"{_article(FAKE_WORDS, rng)} Featured image via Getty Images",
            "subject": "News",
            "date": f"Nov {i % 28 + 1}, 2016",
        })
    return pd.DataFrame(fake), pd.DataFrame(real)


@pytest.fixture
def raw_dir(tmp_path):
    fake, real = make_raw()
    fake.to_csv(tmp_path / "Fake.csv", index=False)
    real.to_csv(tmp_path / "True.csv", index=False)
    return tmp_path


@pytest.fixture
def processed_dir(tmp_path, raw_dir, monkeypatch):
    """Cleaned train/val/test splits written where load_split() looks for them."""
    from fakenews import data

    out = tmp_path / "processed"
    out.mkdir()
    df = data.prepare(data.load_raw(raw_dir), mode="clean")
    for name, part in data.split(df).items():
        part.to_csv(out / f"{name}.csv.gz", index=False)
    monkeypatch.setattr(data, "DATA_PROCESSED", out)
    original = data.load_split
    monkeypatch.setattr(data, "load_split", lambda name, processed_dir=out: original(name, processed_dir))
    return out
