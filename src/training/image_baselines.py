"""Image-only regression baseline using frozen, cached vision embeddings."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.data.loading import load_train
from src.data.quality import leakage_aware_split
from src.evaluation.metrics import regression_metrics
from src.features.image_embeddings import VisionPilotConfig, extract_embeddings, select_pilot_rows


def _record(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _load_cached_embeddings(output_dir: Path, labels: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    manifest = pd.read_csv(output_dir / "manifest.csv")
    embeddings = np.load(output_dir / "embeddings.npy")
    joined = manifest[["sample_id"]].merge(labels[["sample_id", "price"]], on="sample_id", how="inner", validate="one_to_one")
    if joined.empty:
        raise ValueError("No cached embeddings belong to the requested labelled split")
    # Manifest ordering and embedding ordering are identical; use explicit index lookup for safety.
    lookup = {sample_id: index for index, sample_id in enumerate(manifest.sample_id)}
    return embeddings[[lookup[sample_id] for sample_id in joined.sample_id]], joined.price.to_numpy()


def run_image_only_pilot(data_root: str | Path, config_path: str | Path = "configs/image_baseline_pilot.json", results_path: str | Path = "experiments/results/image_baseline_pilot.csv") -> list[dict[str, object]]:
    """Extract a bounded split-aware embedding set and evaluate Ridge image regressors."""
    raw = json.loads(Path(config_path).read_text(encoding="utf-8"))
    raw["data_root"] = Path(data_root)
    raw["output_dir"] = Path(raw["output_dir"])
    train_size = raw.pop("train_sample_size")
    validation_size = raw.pop("validation_sample_size")
    seed, fraction = raw["random_seed"], raw.pop("validation_fraction")
    config = VisionPilotConfig(sample_size=train_size + validation_size, **raw)
    split = leakage_aware_split(load_train(config.data_root), fraction, seed)
    train_rows = select_pilot_rows(split.train, config, train_size).assign(partition="train")
    validation_rows = select_pilot_rows(split.validation, config, validation_size).assign(partition="validation")
    rows = pd.concat([train_rows, validation_rows], ignore_index=True)
    extract_embeddings(rows, config)
    x_train, y_train = _load_cached_embeddings(config.output_dir, split.train)
    x_validation, y_validation = _load_cached_embeddings(config.output_dir, split.validation)
    train_ids = set(split.train.sample_id)
    validation_ids = set(split.validation.sample_id)
    if set(rows[rows.partition == "train"].sample_id) - train_ids or set(rows[rows.partition == "validation"].sample_id) - validation_ids:
        raise RuntimeError("Embedding selection crossed the validation boundary")
    results: list[dict[str, object]] = []
    median_prediction = np.full(len(y_validation), np.median(y_train))
    median_metrics = regression_metrics(y_validation, median_prediction)
    median_result = {"experiment_name": "image_pilot_matched_median", "model": "median", "feature_set": "none", "target_transform": "none", "random_seed": seed, "training_images": len(y_train), "validation_images": len(y_validation), **{f"validation_{name}": value for name, value in median_metrics.items()}, "notes": "Matched 800/200 image-pilot labels; no image embeddings used."}
    results.append(median_result)
    _record(Path(results_path), median_result)
    for transform in ("none", "log1p"):
        target = np.log1p(y_train) if transform == "log1p" else y_train
        model = make_pipeline(StandardScaler(), Ridge(alpha=100.0)).fit(x_train, target)
        prediction = model.predict(x_validation)
        if transform == "log1p":
            prediction = np.expm1(prediction)
        metrics = regression_metrics(y_validation, prediction)
        result = {"experiment_name": f"resnet18_ridge_{transform}", "model": "Ridge(alpha=100)", "feature_set": "resnet18_512d_frozen", "target_transform": transform, "random_seed": seed, "training_images": len(y_train), "validation_images": len(y_validation), **{f"validation_{name}": value for name, value in metrics.items()}, "notes": "Leakage-aware split precedes selection; frozen ResNet-18 image embeddings only."}
        results.append(result)
        _record(Path(results_path), result)
    return results
