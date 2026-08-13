"""Training-only OOF blend-weight selection for the matched late-fusion pilot."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from lightgbm import LGBMRegressor
from sklearn.model_selection import GroupKFold

from src.evaluation.metrics import regression_metrics, smape
from src.features.text import extract_text_features
from src.training.image_baselines import _load_cached_embeddings
from src.training.matched_text_baselines import _matched_rows
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def select_blend_weight(y_true: np.ndarray, text_prediction: np.ndarray, image_prediction: np.ndarray, weights: np.ndarray | None = None) -> tuple[float, float]:
    """Choose a convex blend weight by OOF SMAPE; ties favour the lower text weight."""
    candidates = np.linspace(0.0, 1.0, 21) if weights is None else np.asarray(weights, dtype=float)
    if np.any((candidates < 0) | (candidates > 1)):
        raise ValueError("Fusion weights must be between zero and one")
    scores = [smape(y_true, weight * text_prediction + (1 - weight) * image_prediction) for weight in candidates]
    best = int(np.argmin(scores))
    return float(candidates[best]), float(scores[best])


def _model_predictions(x_image_train, x_text_train, y_train, x_image_eval, x_text_eval, seed: int) -> tuple[np.ndarray, np.ndarray]:
    image_model = make_pipeline(StandardScaler(), Ridge(alpha=100.0)).fit(x_image_train, np.log1p(y_train))
    text_model = LGBMRegressor(n_estimators=300, learning_rate=.05, num_leaves=31, reg_lambda=1., random_state=seed, verbosity=-1, n_jobs=-1).fit(x_text_train, np.log1p(y_train))
    return np.expm1(text_model.predict(x_text_eval)), np.expm1(image_model.predict(x_image_eval))


def _record(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def run_oof_selected_late_fusion(data_root: str | Path, config_path: str | Path = "configs/image_baseline_pilot.json", results_path: str | Path = "experiments/results/oof_late_fusion_pilot.csv") -> list[dict[str, object]]:
    """Select fusion weight using training OOF predictions, then evaluate outer holdout."""
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    train, validation = _matched_rows(Path(data_root), config)
    x_image_train, y_train = _load_cached_embeddings(Path(config["output_dir"]), train)
    x_image_validation, y_validation = _load_cached_embeddings(Path(config["output_dir"]), validation)
    x_text_train = extract_text_features(train.catalog_content)
    x_text_validation = extract_text_features(validation.catalog_content)
    seed = int(config["random_seed"])
    # Image links are unique in this pilot; group exact duplicate catalog text in inner CV.
    groups = train.catalog_content.astype(str).to_numpy()
    splitter = GroupKFold(n_splits=5)
    oof_text, oof_image = np.zeros(len(train)), np.zeros(len(train))
    for fold, (fit_index, holdout_index) in enumerate(splitter.split(x_text_train, y_train, groups)):
        text_prediction, image_prediction = _model_predictions(x_image_train[fit_index], x_text_train.iloc[fit_index], y_train[fit_index], x_image_train[holdout_index], x_text_train.iloc[holdout_index], seed + fold)
        oof_text[holdout_index], oof_image[holdout_index] = text_prediction, image_prediction
    text_weight, oof_score = select_blend_weight(y_train, oof_text, oof_image)
    text_prediction, image_prediction = _model_predictions(x_image_train, x_text_train, y_train, x_image_validation, x_text_validation, seed)
    fusion_prediction = np.clip(text_weight * text_prediction + (1 - text_weight) * image_prediction, 0, None)
    metrics = regression_metrics(y_validation, fusion_prediction)
    row = {"experiment_name":"late_fusion_oof_selected_weight","model":"training_oof_weighted_late_fusion","feature_set":"resnet18_512d + structured_catalog_text_v1","target_transform":"log1p","fusion_weight_text":text_weight,"fusion_weight_image":1-text_weight,"inner_cv_folds":5,"inner_oof_smape":oof_score,"random_seed":seed,"training_products":len(train),"validation_products":len(validation),**{f"validation_{name}":value for name,value in metrics.items()},"notes":"Weight selected from training-only GroupKFold OOF predictions; outer validation was not used for selection."}
    _record(Path(results_path), row)
    return [row]
