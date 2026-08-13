"""Structured-text baselines evaluated on the exact image-pilot sample IDs."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from src.data.loading import load_train
from src.data.quality import leakage_aware_split
from src.evaluation.metrics import regression_metrics
from src.features.text import extract_text_features


def _record(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _matched_rows(data_root: Path, config: dict[str, object]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Join the cached image-pilot manifest to the original duplicate-grouped split."""
    split = leakage_aware_split(load_train(data_root), float(config["validation_fraction"]), int(config["random_seed"]))
    manifest = pd.read_csv(Path(config["output_dir"]) / "manifest.csv", usecols=["sample_id"])
    train = manifest.merge(split.train, on="sample_id", how="inner", validate="one_to_one")
    validation = manifest.merge(split.validation, on="sample_id", how="inner", validate="one_to_one")
    expected_train, expected_validation = int(config["train_sample_size"]), int(config["validation_sample_size"])
    if len(train) != expected_train or len(validation) != expected_validation:
        raise ValueError(f"Expected {expected_train}/{expected_validation} cached split rows; found {len(train)}/{len(validation)}")
    return train, validation


def run_matched_text_baselines(data_root: str | Path, config_path: str | Path = "configs/image_baseline_pilot.json", results_path: str | Path = "experiments/results/matched_text_baselines.csv") -> list[dict[str, object]]:
    """Run text models against the same rows as the cached image pilot."""
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    train, validation = _matched_rows(Path(data_root), config)
    seed = int(config["random_seed"])
    x_train = extract_text_features(train.catalog_content)
    x_validation = extract_text_features(validation.catalog_content)
    y_train, y_validation = train.price.to_numpy(), validation.price.to_numpy()
    output: list[dict[str, object]] = []
    median_prediction = np.full(len(y_validation), np.median(y_train))
    median_metrics = regression_metrics(y_validation, median_prediction)
    baseline = {"experiment_name":"matched_text_median","model":"median","feature_set":"none","target_transform":"none","random_seed":seed,"training_products":len(train),"validation_products":len(validation),**{f"validation_{name}":value for name,value in median_metrics.items()},"notes":"Matched image-pilot products; no text features used."}
    output.append(baseline)
    for transform in ("none", "log1p"):
        target = np.log1p(y_train) if transform == "log1p" else y_train
        model = LGBMRegressor(n_estimators=300, learning_rate=.05, num_leaves=31, reg_lambda=1., random_state=seed, verbosity=-1, n_jobs=-1).fit(x_train, target)
        prediction = model.predict(x_validation)
        if transform == "log1p":
            prediction = np.expm1(prediction)
        metrics = regression_metrics(y_validation, prediction)
        output.append({"experiment_name":f"matched_text_lightgbm_{transform}","model":"LightGBM","feature_set":"structured_catalog_text_v1","target_transform":transform,"random_seed":seed,"training_products":len(train),"validation_products":len(validation),**{f"validation_{name}":value for name,value in metrics.items()},"notes":"Exact same products as the image pilot; original price-scale evaluation."})
    for row in output:
        _record(Path(results_path), row)
    return output
