"""Semantic text-embedding baselines using cached sentence-transformer vectors."""
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
from src.features.text_embeddings import TextEmbeddingConfig, extract_text_embeddings, select_embedding_rows
from src.training.matched_text_baselines import _matched_rows


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
        raise ValueError("No cached text embeddings belong to the requested labelled split")
    lookup = {sample_id: index for index, sample_id in enumerate(manifest.sample_id)}
    return embeddings[[lookup[sample_id] for sample_id in joined.sample_id]], joined.price.to_numpy()


def _resolve_labels(data_root: Path, config: dict[str, object]) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "train_sample_size" in config:
        return _matched_rows(data_root, config)
    split = leakage_aware_split(load_train(data_root), float(config["validation_fraction"]), int(config["random_seed"]))
    return split.train, split.validation


def run_text_embedding_baseline(
    data_root: str | Path,
    config_path: str | Path = "configs/text_embedding_baseline.json",
    results_path: str | Path = "experiments/results/text_embedding_baseline.csv",
) -> list[dict[str, object]]:
    """Extract or reuse cached semantic text embeddings and evaluate LightGBM regressors."""
    raw = json.loads(Path(config_path).read_text(encoding="utf-8"))
    raw["data_root"] = Path(data_root)
    raw["output_dir"] = Path(raw["output_dir"])
    if raw.get("manifest_source"):
        raw["manifest_source"] = Path(raw["manifest_source"])
    seed = int(raw["random_seed"])
    embedding_config = TextEmbeddingConfig(**{key: raw[key] for key in ("data_root", "encoder", "batch_size", "device", "output_dir", "manifest_source") if key in raw})
    rows = select_embedding_rows(load_train(data_root), embedding_config)
    extract_text_embeddings(rows, embedding_config)
    train, validation = _resolve_labels(Path(data_root), raw)
    x_train, y_train = _load_cached_embeddings(embedding_config.output_dir, train)
    x_validation, y_validation = _load_cached_embeddings(embedding_config.output_dir, validation)
    output: list[dict[str, object]] = []
    median_prediction = np.full(len(y_validation), np.median(y_train))
    median_metrics = regression_metrics(y_validation, median_prediction)
    output.append(
        {
            "experiment_name": "semantic_text_median",
            "model": "median",
            "feature_set": "none",
            "target_transform": "none",
            "encoder": raw["encoder"],
            "random_seed": seed,
            "training_products": len(train),
            "validation_products": len(validation),
            **{f"validation_{name}": value for name, value in median_metrics.items()},
            "notes": "Median control on the same labelled products as the semantic text models.",
        }
    )
    for transform in ("none", "log1p"):
        target = np.log1p(y_train) if transform == "log1p" else y_train
        model = LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31, reg_lambda=1.0, random_state=seed, verbosity=-1, n_jobs=-1).fit(x_train, target)
        prediction = model.predict(x_validation)
        if transform == "log1p":
            prediction = np.expm1(prediction)
        metrics = regression_metrics(y_validation, prediction)
        output.append(
            {
                "experiment_name": f"semantic_text_lightgbm_{transform}",
                "model": "LightGBM",
                "feature_set": raw["encoder"],
                "target_transform": transform,
                "encoder": raw["encoder"],
                "random_seed": seed,
                "training_products": len(train),
                "validation_products": len(validation),
                **{f"validation_{name}": value for name, value in metrics.items()},
                "notes": "Frozen semantic text embeddings with LightGBM; evaluated on original price scale.",
            }
        )
    for row in output:
        _record(Path(results_path), row)
    return output
