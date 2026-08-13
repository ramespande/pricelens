"""Reproducible median and structured-feature CPU baseline experiments."""
from __future__ import annotations
import csv, json
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from src.data.loading import load_train
from src.data.quality import leakage_aware_split
from src.evaluation.metrics import regression_metrics
from src.features.text import extract_text_features

def _record(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists: writer.writeheader()
        writer.writerow(row)

def run_baselines(data_root: str | Path, config_path: str | Path = "configs/baseline.json", results_path: str | Path = "experiments/results/baselines.csv") -> list[dict[str, object]]:
    """Run actual baseline experiments and append their validation results."""
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    seed, fraction = config["random_seed"], config["validation_fraction"]
    params = config["model"]
    split = leakage_aware_split(load_train(data_root), fraction, seed)
    y_train, y_valid = split.train.price.to_numpy(), split.validation.price.to_numpy()
    experiments: list[dict[str, object]] = []
    median_prediction = np.full(len(y_valid), np.median(y_train))
    experiments.append({"experiment_name":"median_baseline","model":"median","feature_set":"none","target_transform":"none","random_seed":seed, **{f"validation_{k}":v for k,v in regression_metrics(y_valid, median_prediction).items()}, "notes":"Median of training partition; exact duplicate text/image groups remain within one partition."})
    x_train, x_valid = extract_text_features(split.train.catalog_content), extract_text_features(split.validation.catalog_content)
    try:
        from lightgbm import LGBMRegressor
        estimator = lambda: LGBMRegressor(random_state=seed, verbosity=-1, n_jobs=-1, **params)
        model_name = "LightGBM"
        notes = "CPU LightGBM baseline; predictions clipped to zero; evaluated on original price scale."
    except ImportError:
        estimator = lambda: HistGradientBoostingRegressor(random_state=seed, **params)
        model_name = "HistGradientBoostingRegressor"
        notes = "CPU fallback because LightGBM is unavailable; predictions clipped to zero; evaluated on original price scale."
    for transform in ("none", "log1p"):
        target = np.log1p(y_train) if transform == "log1p" else y_train
        model = estimator().fit(x_train, target)
        prediction = model.predict(x_valid)
        if transform == "log1p": prediction = np.expm1(prediction)
        metrics = regression_metrics(y_valid, prediction)
        experiments.append({"experiment_name":f"{model_name.lower().replace(' ', '_')}_{transform}","model":model_name,"feature_set":"structured_catalog_text_v1","target_transform":transform,"random_seed":seed, **{f"validation_{k}":v for k,v in metrics.items()}, "notes":notes})
    for experiment in experiments: _record(Path(results_path), experiment)
    return experiments
