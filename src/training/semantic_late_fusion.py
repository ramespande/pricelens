"""Semantic late fusion: fixed-weight and OOF-selected blending of cached
semantic text embeddings (MiniLM-L6-v2) and cached image embeddings (ResNet-18).

Both fixed and OOF-selected variants preserve the same experimental protocol as
late_fusion.py and validated_late_fusion.py, substituting semantic text vectors
for the lightweight structured text features used in the original 800/200 pilot.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.evaluation.metrics import regression_metrics, smape
from src.training.matched_text_baselines import _matched_rows


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _record(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _load_embeddings(output_dir: Path, labels: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Load cached embeddings and align to the ordered rows in *labels*.

    The output rows are in the same order as *labels*, not manifest order.
    """
    manifest = pd.read_csv(output_dir / "manifest.csv")
    embeddings = np.load(output_dir / "embeddings.npy")
    # Keep labels as left frame so the output preserves label row order.
    joined = labels[["sample_id", "price"]].merge(
        manifest[["sample_id"]], on="sample_id", how="inner", validate="one_to_one"
    )
    if joined.empty:
        raise ValueError(f"No cached embeddings in {output_dir} match the requested split")
    lookup = {sid: idx for idx, sid in enumerate(manifest.sample_id)}
    return embeddings[[lookup[sid] for sid in joined.sample_id]], joined.price.to_numpy()


def _fit_image_model(x_train: np.ndarray, y_train: np.ndarray) -> object:
    return make_pipeline(StandardScaler(), Ridge(alpha=100.0)).fit(x_train, np.log1p(y_train))


def _fit_text_model(x_train: np.ndarray, y_train: np.ndarray, seed: int) -> object:
    return LGBMRegressor(
        n_estimators=300, learning_rate=0.05, num_leaves=31,
        reg_lambda=1.0, random_state=seed, verbosity=-1, n_jobs=-1,
    ).fit(x_train, np.log1p(y_train))


def _predict_pair(
    image_model, text_model, x_image: np.ndarray, x_text: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    return np.expm1(text_model.predict(x_text)), np.expm1(image_model.predict(x_image))


# ---------------------------------------------------------------------------
# Fixed 50/50 semantic late fusion
# ---------------------------------------------------------------------------

def run_semantic_fixed_late_fusion(
    data_root: str | Path,
    config_path: str | Path = "configs/semantic_fusion_scale_5000.json",
    results_path: str | Path = "experiments/results/semantic_fusion_scale_5000.csv",
) -> list[dict[str, object]]:
    """Train independent log-target models on semantic text and image embeddings;
    evaluate a pre-specified 50/50 blend on the outer holdout.

    The fusion weight is fixed before evaluation; the validation set plays no role
    in weight selection.
    """
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    # Re-use the image-baseline manifest to recover the matched sample IDs.
    image_dir = Path(config["image_output_dir"])
    text_dir = Path(config["text_output_dir"])
    seed = int(config["random_seed"])

    # _matched_rows expects the config to have an "output_dir" pointing at the image manifest.
    image_config = dict(config)
    image_config["output_dir"] = str(image_dir)
    train, validation = _matched_rows(Path(data_root), image_config)

    x_image_train, y_train = _load_embeddings(image_dir, train)
    x_image_val, y_val = _load_embeddings(image_dir, validation)
    x_text_train, _ = _load_embeddings(text_dir, train)
    x_text_val, _ = _load_embeddings(text_dir, validation)

    image_model = _fit_image_model(x_image_train, y_train)
    text_model = _fit_text_model(x_text_train, y_train, seed)
    text_pred, image_pred = _predict_pair(image_model, text_model, x_image_val, x_text_val)
    fusion_pred = np.clip((np.clip(text_pred, 0, None) + np.clip(image_pred, 0, None)) / 2, 0, None)

    candidates = [
        ("semantic_text_log_lgbm", "LightGBM", "minilm_l6_v2_384d_frozen", text_pred, None, None),
        ("resnet18_log_ridge", "Ridge(alpha=100)", "resnet18_512d_frozen", image_pred, None, None),
        ("semantic_fixed_50_50_fusion", "fixed_50_50_late_fusion", "minilm_l6_v2 + resnet18", fusion_pred, 0.5, 0.5),
    ]
    results: list[dict[str, object]] = []
    for name, model, feature_set, pred, wt, wi in candidates:
        metrics = regression_metrics(y_val, pred)
        row: dict[str, object] = {
            "experiment_name": name,
            "model": model,
            "feature_set": feature_set,
            "target_transform": "log1p",
            "fusion_weight_text": wt if wt is not None else "",
            "fusion_weight_image": wi if wi is not None else "",
            "random_seed": seed,
            "training_products": len(train),
            "validation_products": len(validation),
            **{f"validation_{k}": v for k, v in metrics.items()},
            "notes": (
                "Fixed equal-weight semantic fusion; weight pre-specified, validation not used for selection."
                if "fusion" in name else
                "Component model evaluated on the matched scale-5000 sample."
            ),
        }
        results.append(row)
        _record(Path(results_path), row)
    return results


# ---------------------------------------------------------------------------
# OOF-selected semantic late fusion
# ---------------------------------------------------------------------------

def _select_blend_weight(
    y_true: np.ndarray,
    text_pred: np.ndarray,
    image_pred: np.ndarray,
) -> tuple[float, float]:
    """Pick convex blend weight by OOF SMAPE; ties favour lower text weight."""
    candidates = np.linspace(0.0, 1.0, 21)
    scores = [smape(y_true, w * text_pred + (1 - w) * image_pred) for w in candidates]
    best = int(np.argmin(scores))
    return float(candidates[best]), float(scores[best])


def run_semantic_oof_late_fusion(
    data_root: str | Path,
    config_path: str | Path = "configs/semantic_fusion_scale_5000.json",
    results_path: str | Path = "experiments/results/semantic_fusion_scale_5000.csv",
) -> list[dict[str, object]]:
    """Select the semantic fusion weight using training-only GroupKFold OOF predictions,
    then evaluate the outer holdout.

    Validation data plays no role in weight selection; groups are formed from
    exact duplicate catalog text to mirror the leakage-aware outer split.
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

    # Inner 5-fold GroupKFold on training partition only.
    groups = train.catalog_content.astype(str).to_numpy()
    splitter = GroupKFold(n_splits=5)
    oof_text = np.zeros(len(train), dtype=float)
    oof_image = np.zeros(len(train), dtype=float)

    for fold_idx, (fit_idx, holdout_idx) in enumerate(splitter.split(x_text_train, y_train, groups)):
        fold_image_model = _fit_image_model(x_image_train[fit_idx], y_train[fit_idx])
        fold_text_model = _fit_text_model(x_text_train[fit_idx], y_train[fit_idx], seed + fold_idx)
        oof_text[holdout_idx], oof_image[holdout_idx] = _predict_pair(
            fold_image_model, fold_text_model,
            x_image_train[holdout_idx], x_text_train[holdout_idx],
        )

    text_weight, oof_smape = _select_blend_weight(y_train, oof_text, oof_image)

    # Refit on full training partition, then evaluate outer holdout.
    image_model = _fit_image_model(x_image_train, y_train)
    text_model = _fit_text_model(x_text_train, y_train, seed)
    text_pred, image_pred = _predict_pair(image_model, text_model, x_image_val, x_text_val)
    fusion_pred = np.clip(text_weight * text_pred + (1 - text_weight) * image_pred, 0, None)

    metrics = regression_metrics(y_val, fusion_pred)
    row: dict[str, object] = {
        "experiment_name": "semantic_oof_selected_fusion",
        "model": "training_oof_weighted_late_fusion",
        "feature_set": "minilm_l6_v2 + resnet18",
        "target_transform": "log1p",
        "fusion_weight_text": text_weight,
        "fusion_weight_image": round(1 - text_weight, 10),
        "inner_cv_folds": 5,
        "inner_oof_smape": oof_smape,
        "random_seed": seed,
        "training_products": len(train),
        "validation_products": len(validation),
        **{f"validation_{k}": v for k, v in metrics.items()},
        "notes": (
            "OOF weight selected from training-only GroupKFold; outer validation not used for selection."
        ),
    }
    _record(Path(results_path), row)
    return [row]
