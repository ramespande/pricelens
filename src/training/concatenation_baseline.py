"""Concatenation (early) fusion baseline.

Horizontally stacks the cached semantic text embedding (384-D MiniLM-L6-v2)
and the cached image embedding (512-D ResNet-18) into a single 896-D joint
feature vector, then fits a single LightGBM regressor on the concatenated
representation. This is compared against late-fusion variants under the same
matched 5K experimental protocol.

No validation data is used during training or feature construction.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from src.evaluation.metrics import regression_metrics
from src.training.matched_text_baselines import _matched_rows
from src.training.semantic_late_fusion import _load_embeddings


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _record(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def concatenate_embeddings(text: np.ndarray, image: np.ndarray) -> np.ndarray:
    """Horizontally stack text and image embedding matrices.

    Args:
        text:  (n, d_text) array of semantic text embeddings.
        image: (n, d_image) array of image embeddings.

    Returns:
        (n, d_text + d_image) joint feature matrix.
    """
    if text.shape[0] != image.shape[0]:
        raise ValueError(
            f"Text ({text.shape[0]}) and image ({image.shape[0]}) embedding counts must match"
        )
    return np.concatenate([text, image], axis=1)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_concatenation_baseline(
    data_root: str | Path,
    config_path: str | Path = "configs/concatenation_baseline_scale_5000.json",
    results_path: str | Path = "experiments/results/concatenation_baseline_scale_5000.csv",
) -> list[dict[str, object]]:
    """Fit a single LightGBM on the joint [text; image] embedding and evaluate on
    the matched scale-5000 holdout.

    Both direct-price and log-target variants are evaluated, mirroring the
    training protocol used in the other baseline experiments.
    """
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    image_dir = Path(config["image_output_dir"])
    text_dir = Path(config["text_output_dir"])
    seed = int(config["random_seed"])

    image_config = dict(config)
    image_config["output_dir"] = str(image_dir)
    train, validation = _matched_rows(Path(data_root), image_config)

    x_image_train, y_train = _load_embeddings(image_dir, train)
    x_image_val, y_val = _load_embeddings(image_dir, validation)
    x_text_train, _ = _load_embeddings(text_dir, train)
    x_text_val, _ = _load_embeddings(text_dir, validation)

    x_train = concatenate_embeddings(x_text_train, x_image_train)
    x_val = concatenate_embeddings(x_text_val, x_image_val)

    results: list[dict[str, object]] = []

    # Median control (same labels, no features)
    median_pred = np.full(len(y_val), np.median(y_train))
    median_metrics = regression_metrics(y_val, median_pred)
    median_row: dict[str, object] = {
        "experiment_name": "concat_fusion_median",
        "model": "median",
        "feature_set": "none",
        "target_transform": "none",
        "input_dimension": int(x_train.shape[1]),
        "random_seed": seed,
        "training_products": len(train),
        "validation_products": len(validation),
        **{f"validation_{k}": v for k, v in median_metrics.items()},
        "notes": "Matched scale-5000 median control; no features used.",
    }
    results.append(median_row)
    _record(Path(results_path), median_row)

    for transform in ("none", "log1p"):
        target = np.log1p(y_train) if transform == "log1p" else y_train
        model = LGBMRegressor(
            n_estimators=300, learning_rate=0.05, num_leaves=31,
            reg_lambda=1.0, random_state=seed, verbosity=-1, n_jobs=-1,
        ).fit(x_train, target)
        pred = model.predict(x_val)
        if transform == "log1p":
            pred = np.expm1(pred)
        metrics = regression_metrics(y_val, pred)
        row: dict[str, object] = {
            "experiment_name": f"concat_fusion_lgbm_{transform}",
            "model": "LightGBM",
            "feature_set": "minilm_l6_v2_384d + resnet18_512d",
            "target_transform": transform,
            "input_dimension": int(x_train.shape[1]),
            "random_seed": seed,
            "training_products": len(train),
            "validation_products": len(validation),
            **{f"validation_{k}": v for k, v in metrics.items()},
            "notes": (
                "Early/concatenation fusion: 896-D joint embedding [text; image] "
                "fitted with a single LightGBM; no fusion weight tuning required."
            ),
        }
        results.append(row)
        _record(Path(results_path), row)

    return results
