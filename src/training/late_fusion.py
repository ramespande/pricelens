"""Fixed-weight late fusion for the matched text/image pilot."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.evaluation.metrics import regression_metrics
from src.features.text import extract_text_features
from src.training.image_baselines import _load_cached_embeddings
from src.training.matched_text_baselines import _matched_rows


def equal_weight_late_fusion(text_prediction: np.ndarray, image_prediction: np.ndarray) -> np.ndarray:
    """Return a fixed 50/50 original-price blend without validation-set tuning."""
    text = np.asarray(text_prediction, dtype=float)
    image = np.asarray(image_prediction, dtype=float)
    if text.shape != image.shape:
        raise ValueError("Text and image predictions must have the same shape")
    return np.clip((np.clip(text, 0, None) + np.clip(image, 0, None)) / 2, 0, None)


def _record(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def run_fixed_late_fusion(data_root: str | Path, config_path: str | Path = "configs/image_baseline_pilot.json", results_path: str | Path = "experiments/results/late_fusion_pilot.csv") -> list[dict[str, object]]:
    """Train independent log-target models and evaluate a pre-specified 50/50 blend."""
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    train, validation = _matched_rows(Path(data_root), config)
    cache_dir = Path(config["output_dir"])
    x_image_train, y_train = _load_cached_embeddings(cache_dir, train)
    x_image_validation, y_validation = _load_cached_embeddings(cache_dir, validation)
    x_text_train = extract_text_features(train.catalog_content)
    x_text_validation = extract_text_features(validation.catalog_content)
    seed = int(config["random_seed"])
    image_model = make_pipeline(StandardScaler(), Ridge(alpha=100.0)).fit(x_image_train, np.log1p(y_train))
    text_model = LGBMRegressor(n_estimators=300, learning_rate=.05, num_leaves=31, reg_lambda=1., random_state=seed, verbosity=-1, n_jobs=-1).fit(x_text_train, np.log1p(y_train))
    image_prediction = np.expm1(image_model.predict(x_image_validation))
    text_prediction = np.expm1(text_model.predict(x_text_validation))
    fusion_prediction = equal_weight_late_fusion(text_prediction, image_prediction)
    candidates = [("image_log_ridge", "Ridge(alpha=100)", "resnet18_512d_frozen", image_prediction), ("text_log_lightgbm", "LightGBM", "structured_catalog_text_v1", text_prediction), ("late_fusion_equal_weight", "fixed_50_50_late_fusion", "resnet18_512d + structured_catalog_text_v1", fusion_prediction)]
    results: list[dict[str, object]] = []
    for name, model, feature_set, prediction in candidates:
        metrics = regression_metrics(y_validation, prediction)
        row = {"experiment_name":name,"model":model,"feature_set":feature_set,"target_transform":"log1p","fusion_weight_text":0.5 if name == "late_fusion_equal_weight" else "","fusion_weight_image":0.5 if name == "late_fusion_equal_weight" else "","random_seed":seed,"training_products":len(train),"validation_products":len(validation),**{f"validation_{metric}":value for metric,value in metrics.items()},"notes":"Fixed equal-weight late fusion was pre-specified; validation was not used to select its weight." if name == "late_fusion_equal_weight" else "Component model evaluated on the matched pilot."}
        results.append(row)
        _record(Path(results_path), row)
    return results
