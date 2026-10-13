"""Paths and constants shared across the project."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"

SEED = 42
LABELS = {0: "real", 1: "fake"}  # "fake" is the positive class everywhere


def ensure_dirs() -> None:
    for d in (DATA_PROCESSED, MODELS_DIR, RESULTS_DIR, PREDICTIONS_DIR):
        d.mkdir(parents=True, exist_ok=True)
