"""Loading, leakage cleaning and splitting of the ISOT Fake and Real News dataset.

The raw dataset (Kaggle: clmentbisaillon/fake-and-real-news-dataset) is known to be
"too easy": nearly every real article comes from Reuters and starts with a dateline such as
"WASHINGTON (Reuters) - ", while fake articles carry their own source artifacts
("Featured image via ...", "21st Century Wire", "(VIDEO)" tags in titles). A model can reach
~99% accuracy by spotting these markers without reading the content.

`clean_text` / `clean_title` strip those markers so the benchmark measures whether a model
detects fake news from content and style, not from where the article was scraped.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import DATA_PROCESSED, DATA_RAW, SEED, ensure_dirs

# --- leakage patterns -------------------------------------------------------------------

# "WASHINGTON (Reuters) - ", "LONDON/PARIS (Reuters) - ", "(Reuters) - "
_DATELINE = re.compile(r"^\s*[^\n]{0,120}?\(\s*Reuters\s*\)\s*[-–—]\s*", re.IGNORECASE)
_REUTERS = re.compile(r"\breuters\b", re.IGNORECASE)
# Trailing image / photo credits typical of the fake sources
_IMAGE_CREDIT = re.compile(
    r"(featured\s+image|photo|image|screenshot|screen\s+capture)\s+(via|by|credit|courtesy\s+of)\b[^\n]*$",
    re.IGNORECASE,
)
_SOURCE_NAMES = re.compile(r"\b(21st\s+century\s+wire|21wire|getty\s+images|shutterstock)\b", re.IGNORECASE)
_URL = re.compile(r"(https?://\S+|www\.\S+|\b\S+\.(com|org|net|co)/\S*)", re.IGNORECASE)
_HANDLE = re.compile(r"@\w+")
_PIC_TWITTER = re.compile(r"pic\.twitter\.com/\S+", re.IGNORECASE)
# Title tags: (VIDEO), [VIDEO], (TWEETS), WATCH:, BREAKING:
_TITLE_TAGS = re.compile(
    r"[\(\[]\s*(video|videos|tweet|tweets|image|images|photo|photos|details|screenshots?)\s*[\)\]]"
    r"|\b(watch|breaking)\s*:",
    re.IGNORECASE,
)
_WS = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Remove source artifacts that leak the label from an article body."""
    if not isinstance(text, str):
        return ""
    t = _DATELINE.sub("", text)
    t = _PIC_TWITTER.sub(" ", t)
    t = _URL.sub(" ", t)
    t = _HANDLE.sub(" ", t)
    t = _IMAGE_CREDIT.sub(" ", t.strip())
    t = _REUTERS.sub(" ", t)
    t = _SOURCE_NAMES.sub(" ", t)
    return _WS.sub(" ", t).strip()


def clean_title(title: str) -> str:
    """Remove source tags from a headline. Casing is kept and handled by each model."""
    if not isinstance(title, str):
        return ""
    t = _TITLE_TAGS.sub(" ", title)
    t = _REUTERS.sub(" ", t)
    return _WS.sub(" ", t).strip(" -–—:|").strip()


# --- loading -----------------------------------------------------------------------------


def load_raw(data_dir: Path = DATA_RAW) -> pd.DataFrame:
    """Read Fake.csv and True.csv and return one frame with label 1 = fake, 0 = real."""
    data_dir = Path(data_dir)
    fake_path, true_path = data_dir / "Fake.csv", data_dir / "True.csv"
    missing = [p.name for p in (fake_path, true_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing {', '.join(missing)} in {data_dir}. Download the dataset with "
            "`kaggle datasets download -d clmentbisaillon/fake-and-real-news-dataset -p data/raw --unzip`."
        )
    fake = pd.read_csv(fake_path).assign(label=1)
    real = pd.read_csv(true_path).assign(label=0)
    df = pd.concat([fake, real], ignore_index=True)
    df["date"] = pd.to_datetime(df["date"].astype(str).str.strip(), errors="coerce", format="mixed")
    return df[["title", "text", "subject", "date", "label"]]


def prepare(df: pd.DataFrame, mode: str = "clean", min_chars: int = 200) -> pd.DataFrame:
    """Build the model input column.

    mode="clean": strip leakage artifacts (default, used for all reported results).
    mode="raw":   keep text as scraped; only used by the leakage audit.

    The `subject` column is always dropped: its values are disjoint between classes
    (e.g. "politicsNews" only exists for real articles), so it is a pure label leak.
    """
    if mode not in {"clean", "raw"}:
        raise ValueError("mode must be 'clean' or 'raw'")
    out = df.copy()
    if mode == "clean":
        out["title"] = out["title"].map(clean_title)
        out["text"] = out["text"].map(clean_text)
    else:
        out["title"] = out["title"].fillna("").astype(str).str.strip()
        out["text"] = out["text"].fillna("").astype(str).str.strip()
    # Many fake rows have an empty body (title only); keep articles with real content.
    out = out[out["text"].str.len() >= min_chars]
    # The dataset contains many duplicated articles; duplicates across splits inflate scores.
    out = out.drop_duplicates(subset="text").drop_duplicates(subset="title")
    out["input"] = out["title"] + ". " + out["text"]
    out = out.drop(columns=["subject"]).reset_index(drop=True)
    out.insert(0, "id", range(len(out)))
    return out


def split(
    df: pd.DataFrame, strategy: str = "random", test_size: float = 0.15, val_size: float = 0.15, seed: int = SEED
) -> dict[str, pd.DataFrame]:
    """Split into train / val / test.

    strategy="random":   stratified random split.
    strategy="temporal": train on the oldest articles, test on the most recent ones
                         (closer to deployment, where the model meets future news).
    """
    if strategy == "random":
        train, rest = train_test_split(df, test_size=test_size + val_size, stratify=df["label"], random_state=seed)
        val, test = train_test_split(
            rest, test_size=test_size / (test_size + val_size), stratify=rest["label"], random_state=seed
        )
    elif strategy == "temporal":
        dated = df.dropna(subset=["date"]).sort_values("date")
        n = len(dated)
        n_train = int(n * (1 - test_size - val_size))
        n_val = int(n * val_size)
        train, val, test = dated.iloc[:n_train], dated.iloc[n_train : n_train + n_val], dated.iloc[n_train + n_val :]
    else:
        raise ValueError("strategy must be 'random' or 'temporal'")
    return {"train": train.reset_index(drop=True), "val": val.reset_index(drop=True), "test": test.reset_index(drop=True)}


def load_split(name: str, processed_dir: Path = DATA_PROCESSED) -> pd.DataFrame:
    path = Path(processed_dir) / f"{name}.csv.gz"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python -m fakenews.data` first.")
    return pd.read_csv(path, parse_dates=["date"]).fillna({"title": "", "text": "", "input": ""})


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean and split the ISOT fake news dataset.")
    parser.add_argument("--data-dir", type=Path, default=DATA_RAW)
    parser.add_argument("--strategy", choices=["random", "temporal"], default="random")
    args = parser.parse_args()

    ensure_dirs()
    raw = load_raw(args.data_dir)
    df = prepare(raw, mode="clean")
    splits = split(df, strategy=args.strategy)
    for name, part in splits.items():
        part.to_csv(DATA_PROCESSED / f"{name}.csv.gz", index=False)

    print(f"Raw articles: {len(raw):,} -> after cleaning and deduplication: {len(df):,}")
    for name, part in splits.items():
        print(f"  {name:5s} {len(part):6,}  fake share {part['label'].mean():.1%}")


if __name__ == "__main__":
    main()
