import pandas as pd
import pytest

from fakenews.data import clean_text, clean_title, load_raw, prepare, split


@pytest.mark.parametrize("raw, expected", [
    ("WASHINGTON (Reuters) - The Senate voted.", "The Senate voted."),
    ("LONDON/PARIS (Reuters) - Talks resumed.", "Talks resumed."),
    ("(Reuters) - A spokesman said.", "A spokesman said."),
    ("He said it. Featured image via Getty Images", "He said it."),
    ("Read more at https://example.com/story now", "Read more at now"),
    ("As @realDonaldTrump wrote pic.twitter.com/abc123", "As wrote"),
    ("A 21st Century Wire report.", "A report."),
])
def test_clean_text_removes_source_artifacts(raw, expected):
    assert clean_text(raw) == expected


def test_clean_text_keeps_content_and_handles_missing():
    assert clean_text("The economy grew 3% last quarter.") == "The economy grew 3% last quarter."
    assert clean_text(None) == ""


@pytest.mark.parametrize("raw, expected", [
    ("Trump Speaks Out (VIDEO)", "Trump Speaks Out"),
    ("WATCH: Senator loses it [Video]", "Senator loses it"),
    ("BREAKING: Big news (TWEETS)", "Big news"),
])
def test_clean_title(raw, expected):
    assert clean_title(raw) == expected


def test_load_raw_labels_and_columns(raw_dir):
    df = load_raw(raw_dir)
    assert set(df["label"]) == {0, 1}
    assert list(df.columns) == ["title", "text", "subject", "date", "label"]
    assert df["date"].notna().all()


def test_load_raw_missing_files(tmp_path):
    with pytest.raises(FileNotFoundError, match="kaggle datasets download"):
        load_raw(tmp_path)


def test_prepare_clean_has_no_leaky_markers(raw_dir):
    df = prepare(load_raw(raw_dir), mode="clean")
    joined = " ".join(df["input"]).lower()
    for marker in ("reuters", "featured image", "getty", "(video)"):
        assert marker not in joined
    assert "subject" not in df.columns


def test_prepare_raw_keeps_markers_and_dedupes(raw_dir):
    raw = load_raw(raw_dir)
    doubled = pd.concat([raw, raw])
    df = prepare(doubled, mode="raw")
    assert "reuters" in " ".join(df["input"]).lower()
    assert df["text"].is_unique and df["title"].is_unique


def test_prepare_drops_short_articles(raw_dir):
    raw = load_raw(raw_dir)
    raw.loc[0, "text"] = "too short"
    df = prepare(raw)
    assert len(df) == len(raw) - 1
    assert "too short" not in set(df["text"])


@pytest.mark.parametrize("strategy", ["random", "temporal"])
def test_split_is_disjoint_and_complete(raw_dir, strategy):
    df = prepare(load_raw(raw_dir))
    parts = split(df, strategy=strategy)
    ids = [set(p["id"]) for p in parts.values()]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
    assert sum(len(i) for i in ids) == len(df)
    if strategy == "random":
        for p in parts.values():
            assert abs(p["label"].mean() - df["label"].mean()) < 0.1
    else:
        assert parts["train"]["date"].max() <= parts["test"]["date"].min()
