"""Data quality reporting and leakage-aware splitting."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split


def data_quality_report(frame: pd.DataFrame) -> dict[str, Any]:
    """Return serializable missingness, duplicate, type, and target summaries."""
    report: dict[str, Any] = {
        "rows": len(frame),
        "columns": list(frame.columns),
        "dtypes": {column: str(dtype) for column, dtype in frame.dtypes.items()},
        "missing_values": {column: int(value) for column, value in frame.isna().sum().items()},
        "duplicate_rows": int(frame.duplicated().sum()),
    }
    for column in ("sample_id", "catalog_content", "image_link"):
        if column in frame:
            report[f"duplicate_{column}"] = int(frame[column].duplicated().sum())
            report[f"unique_{column}"] = int(frame[column].nunique(dropna=False))
    if "price" in frame:
        report["price_statistics"] = {
            str(key): float(value)
            for key, value in frame["price"].describe(percentiles=[.01, .05, .25, .5, .75, .95, .99]).items()
        }
    return report


@dataclass(frozen=True)
class SplitResult:
    train: pd.DataFrame
    validation: pd.DataFrame
    held_out_duplicate_rows: int


def leakage_aware_split(frame: pd.DataFrame, validation_fraction: float = 0.2, random_seed: int = 42) -> SplitResult:
    """Split rows while grouping exact duplicate catalog text or image links.

    Each connected duplicate group (sharing catalog_content or image_link) stays in one
    partition. This avoids direct, exact-duplicate leakage without attempting semantic
    deduplication. Groups are assigned with deterministic shuffled greedy balancing.
    """
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    required = {"catalog_content", "image_link"}
    if missing := required.difference(frame.columns):
        raise ValueError(f"Cannot create leakage-aware split; missing {sorted(missing)}")
    # Build connected components using duplicate identifiers via a compact union-find.
    parent = list(range(len(frame)))
    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb: parent[rb] = ra
    for column in ("catalog_content", "image_link"):
        first: dict[object, int] = {}
        for pos, value in enumerate(frame[column].tolist()):
            if value in first: union(pos, first[value])
            else: first[value] = pos
    groups: dict[int, list[int]] = {}
    for pos in range(len(frame)):
        groups.setdefault(find(pos), []).append(pos)
    group_rows = list(groups.values())
    order = pd.Series(range(len(group_rows))).sample(frac=1, random_state=random_seed).tolist()
    target = round(len(frame) * validation_fraction)
    validation_positions: list[int] = []
    for group_index in order:
        group = group_rows[group_index]
        if len(validation_positions) < target and abs(target - (len(validation_positions) + len(group))) <= abs(target - len(validation_positions)):
            validation_positions.extend(group)
    validation_set = set(validation_positions)
    train_positions = [i for i in range(len(frame)) if i not in validation_set]
    train = frame.iloc[train_positions].copy().reset_index(drop=True)
    validation = frame.iloc[validation_positions].copy().reset_index(drop=True)
    overlap = (set(train["catalog_content"]) & set(validation["catalog_content"])) | (set(train["image_link"]) & set(validation["image_link"]))
    if overlap:
        raise RuntimeError("Exact duplicate leakage detected after split")
    return SplitResult(train=train, validation=validation, held_out_duplicate_rows=len(validation))
