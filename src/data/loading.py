"""Dataset path resolution and schema-aware CSV loading."""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

REQUIRED_TRAIN_COLUMNS = {"sample_id", "catalog_content", "image_link", "price"}
REQUIRED_TEST_COLUMNS = {"sample_id", "catalog_content", "image_link"}


def resolve_data_root(data_root: str | Path | None = None) -> Path:
    """Return the dataset root from an argument or ``DATA_ROOT`` environment variable."""
    value = data_root or os.environ.get("DATA_ROOT")
    if not value:
        raise ValueError("Dataset root is required. Set DATA_ROOT or pass data_root.")
    root = Path(value).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")
    return root


def validate_schema(frame: pd.DataFrame, *, split: str) -> None:
    """Validate required columns and the numeric train target."""
    expected = REQUIRED_TRAIN_COLUMNS if split == "train" else REQUIRED_TEST_COLUMNS
    missing = expected.difference(frame.columns)
    if missing:
        raise ValueError(f"{split}.csv is missing required columns: {sorted(missing)}")
    if split == "train" and not pd.api.types.is_numeric_dtype(frame["price"]):
        raise TypeError("train.csv column 'price' must be numeric")


def load_split(data_root: str | Path | None, split: str) -> pd.DataFrame:
    """Load and validate ``train.csv`` or ``test.csv`` without modifying the source data."""
    if split not in {"train", "test"}:
        raise ValueError("split must be 'train' or 'test'")
    path = resolve_data_root(data_root) / f"{split}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Expected file not found: {path}")
    frame = pd.read_csv(path)
    validate_schema(frame, split=split)
    return frame


def load_train(data_root: str | Path | None = None) -> pd.DataFrame:
    return load_split(data_root, "train")


def load_test(data_root: str | Path | None = None) -> pd.DataFrame:
    return load_split(data_root, "test")
