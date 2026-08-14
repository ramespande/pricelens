"""Adaptive (gated) multimodal fusion.

A shallow MLP gating network learns per-sample fusion weights from the
concatenated [text_emb; image_emb] feature vector. For each sample it outputs
a scalar gate g ∈ (0, 1), so the final prediction is:

    price = g * text_price_pred + (1 - g) * image_price_pred

The gate is trained to minimise the mean-squared error of the fused log-price
prediction, using the independently pre-trained log-target component models as
frozen predictors.

Architecture (CPU-friendly, small matched set):
  - Input:  d_text + d_image  (default 384 + 512 = 896)
  - Hidden: 128 units, ReLU
  - Output: 1 unit, Sigmoid

Training uses Adam (lr=1e-3), weight-decay=1e-4, 200 epochs, batch_size=256.
Early stopping is applied on a 20% inner hold-out (split inside training only).

The gate is fitted entirely within the training partition using a single
training/inner-validation split. The outer validation holdout is not used
during gate training or model selection.
"""
from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.evaluation.metrics import regression_metrics
from src.training.matched_text_baselines import _matched_rows
from src.training.semantic_late_fusion import _load_embeddings

LOGGER = logging.getLogger(__name__)


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


def _fit_image_model(x: np.ndarray, y: np.ndarray) -> object:
    return make_pipeline(StandardScaler(), Ridge(alpha=100.0)).fit(x, np.log1p(y))


def _fit_text_model(x: np.ndarray, y: np.ndarray, seed: int) -> object:
    return LGBMRegressor(
        n_estimators=300, learning_rate=0.05, num_leaves=31,
        reg_lambda=1.0, random_state=seed, verbosity=-1, n_jobs=-1,
    ).fit(x, np.log1p(y))


# ---------------------------------------------------------------------------
# MLP Gate (pure-NumPy, no torch dependency)
# ---------------------------------------------------------------------------

class _AdaptiveGate:
    """Shallow MLP gate: Linear(d_in, 128) → ReLU → Linear(128, 1) → Sigmoid.

    Trained with mini-batch Adam to minimise MSE of the gated log-price fusion.
    Uses pure NumPy so no new heavy dependency is introduced for this step.
    """

    def __init__(self, d_in: int, hidden: int = 128, seed: int = 42) -> None:
        rng = np.random.default_rng(seed)
        scale1 = np.sqrt(2.0 / d_in)
        self.W1 = rng.normal(0, scale1, (d_in, hidden)).astype(np.float32)
        self.b1 = np.zeros(hidden, dtype=np.float32)
        scale2 = np.sqrt(2.0 / hidden)
        self.W2 = rng.normal(0, scale2, (hidden, 1)).astype(np.float32)
        self.b2 = np.zeros(1, dtype=np.float32)
        # Adam state
        self._m = [np.zeros_like(p) for p in (self.W1, self.b1, self.W2, self.b2)]
        self._v = [np.zeros_like(p) for p in (self.W1, self.b1, self.W2, self.b2)]
        self._t = 0

    def _params(self) -> list[np.ndarray]:
        return [self.W1, self.b1, self.W2, self.b2]

    def forward(self, x: np.ndarray) -> np.ndarray:
        h = np.maximum(0, x @ self.W1 + self.b1)           # ReLU
        logit = h @ self.W2 + self.b2                       # (n, 1)
        return 1.0 / (1.0 + np.exp(-logit))                # Sigmoid → (n, 1)

    def fit(
        self,
        x: np.ndarray,
        log_text: np.ndarray,
        log_image: np.ndarray,
        log_price: np.ndarray,
        epochs: int = 200,
        batch_size: int = 256,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        patience: int = 20,
        val_fraction: float = 0.15,
        seed: int = 42,
    ) -> dict[str, object]:
        """Fit the gate to minimise MSE of gated log-price prediction.

        Args:
            x:          (n, d) concatenated [text; image] features (standardised).
            log_text:   (n,) log-price predictions from the text model.
            log_image:  (n,) log-price predictions from the image model.
            log_price:  (n,) true log-prices.

        Returns:
            dict with training history keys (best_val_loss, stopped_epoch).
        """
        n = len(x)
        idx = np.random.default_rng(seed).permutation(n)
        n_val = max(1, int(n * val_fraction))
        val_idx, fit_idx = idx[:n_val], idx[n_val:]

        x_fit, x_val = x[fit_idx].astype(np.float32), x[val_idx].astype(np.float32)
        lt_fit, lt_val = log_text[fit_idx], log_text[val_idx]
        li_fit, li_val = log_image[fit_idx], log_image[val_idx]
        lp_fit, lp_val = log_price[fit_idx], log_price[val_idx]

        best_val_loss = np.inf
        best_params = [p.copy() for p in self._params()]
        no_improve = 0

        for epoch in range(epochs):
            perm = np.random.default_rng(seed + epoch).permutation(len(x_fit))
            for start in range(0, len(x_fit), batch_size):
                b = perm[start: start + batch_size]
                xb = x_fit[b]
                lt_b, li_b, lp_b = lt_fit[b], li_fit[b], lp_fit[b]

                # Forward
                h = np.maximum(0, xb @ self.W1 + self.b1)
                gate = 1.0 / (1.0 + np.exp(-(h @ self.W2 + self.b2)))  # (bs, 1)
                g = gate[:, 0]
                pred = g * lt_b + (1 - g) * li_b                        # (bs,)
                residual = pred - lp_b                                   # (bs,)

                # Backward through gate
                d_pred = 2.0 * residual / len(b)
                d_g = d_pred * (lt_b - li_b)                            # (bs,)
                d_gate = d_g[:, None] * gate * (1 - gate)               # (bs, 1)
                d_W2 = h.T @ d_gate + weight_decay * self.W2
                d_b2 = d_gate.sum(axis=0) + weight_decay * self.b2
                d_h = (d_gate @ self.W2.T) * (h > 0)
                d_W1 = xb.T @ d_h + weight_decay * self.W1
                d_b1 = d_h.sum(axis=0) + weight_decay * self.b1

                grads = [d_W1, d_b1, d_W2, d_b2]
                self._t += 1
                beta1, beta2, eps = 0.9, 0.999, 1e-8
                for i, (p, g_arr) in enumerate(zip(self._params(), grads)):
                    self._m[i] = beta1 * self._m[i] + (1 - beta1) * g_arr
                    self._v[i] = beta2 * self._v[i] + (1 - beta2) * g_arr ** 2
                    m_hat = self._m[i] / (1 - beta1 ** self._t)
                    v_hat = self._v[i] / (1 - beta2 ** self._t)
                    p -= lr * m_hat / (np.sqrt(v_hat) + eps)

            # Validation loss
            gate_val = self.forward(x_val)[:, 0]
            val_pred = gate_val * lt_val + (1 - gate_val) * li_val
            val_loss = float(np.mean((val_pred - lp_val) ** 2))
            if val_loss < best_val_loss - 1e-6:
                best_val_loss = val_loss
                best_params = [p.copy() for p in self._params()]
                no_improve = 0
            else:
                no_improve += 1
            if no_improve >= patience:
                LOGGER.debug("Early stopping at epoch %d; best val MSE %.5f", epoch, best_val_loss)
                break

        # Restore best weights
        self.W1, self.b1, self.W2, self.b2 = best_params
        return {"best_val_mse": best_val_loss, "stopped_epoch": epoch}

    def predict_gate(self, x: np.ndarray) -> np.ndarray:
        """Return gate values g ∈ (0, 1) for each sample."""
        return self.forward(x.astype(np.float32))[:, 0]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_adaptive_fusion(
    data_root: str | Path,
    config_path: str | Path = "configs/semantic_fusion_scale_5000.json",
    results_path: str | Path = "experiments/results/adaptive_fusion_scale_5000.csv",
    epochs: int = 200,
    hidden: int = 128,
    batch_size: int = 256,
) -> list[dict[str, object]]:
    """Fit an MLP gate on the training partition and evaluate on the outer holdout.

    Training steps:
      1. Load cached semantic text and image embeddings.
      2. Fit frozen Ridge (image) and LightGBM (text) log-target component models.
      3. Standardise the concatenated [text; image] embedding as input to the gate.
      4. Train the MLP gate on the training partition using a 15% inner holdout
         for early stopping (no outer validation data is used).
      5. Evaluate the gated prediction on the outer holdout.
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

    # Use out-of-fold component predictions for gate supervision. In-sample
    # component predictions would permit the gate to exploit their overfitting.
    groups = train.catalog_content.astype(str).to_numpy()
    splitter = GroupKFold(n_splits=5)
    log_text_oof = np.zeros(len(train), dtype=float)
    log_image_oof = np.zeros(len(train), dtype=float)
    for fold, (fit_idx, holdout_idx) in enumerate(splitter.split(x_text_train, y_train, groups)):
        fold_image = _fit_image_model(x_image_train[fit_idx], y_train[fit_idx])
        fold_text = _fit_text_model(x_text_train[fit_idx], y_train[fit_idx], seed + fold)
        log_text_oof[holdout_idx] = fold_text.predict(x_text_train[holdout_idx])
        log_image_oof[holdout_idx] = fold_image.predict(x_image_train[holdout_idx])

    # Refit component models on all outer-training rows for outer validation.
    image_model = _fit_image_model(x_image_train, y_train)
    text_model = _fit_text_model(x_text_train, y_train, seed)
    log_text_val = text_model.predict(x_text_val)
    log_image_val = image_model.predict(x_image_val)
    log_y_train = np.log1p(y_train)

    # Concatenate embeddings and standardise for the gate network
    from numpy.linalg import norm  # noqa: F401
    x_joint_train = np.concatenate([x_text_train, x_image_train], axis=1)
    x_joint_val = np.concatenate([x_text_val, x_image_val], axis=1)
    scaler = StandardScaler().fit(x_joint_train)
    x_joint_train_s = scaler.transform(x_joint_train).astype(np.float32)
    x_joint_val_s = scaler.transform(x_joint_val).astype(np.float32)

    gate = _AdaptiveGate(d_in=x_joint_train_s.shape[1], hidden=hidden, seed=seed)
    history = gate.fit(
        x_joint_train_s, log_text_oof, log_image_oof, log_y_train,
        epochs=epochs, batch_size=batch_size, seed=seed,
    )

    # Evaluate on outer holdout
    gate_val = gate.predict_gate(x_joint_val_s)
    fused_log_val = gate_val * log_text_val + (1 - gate_val) * log_image_val
    pred = np.clip(np.expm1(fused_log_val), 0, None)
    metrics = regression_metrics(y_val, pred)

    gate_train = gate.predict_gate(x_joint_train_s)
    row: dict[str, object] = {
        "experiment_name": "adaptive_mlp_gate_fusion",
        "model": f"MLP_gate(hidden={hidden})+Ridge+LightGBM",
        "feature_set": "minilm_l6_v2_384d + resnet18_512d (gate input)",
        "target_transform": "log1p",
        "gate_hidden_units": hidden,
        "gate_epochs": epochs,
        "gate_batch_size": batch_size,
        "gate_mean_train": float(np.mean(gate_train)),
        "gate_std_train": float(np.std(gate_train)),
        "gate_mean_val": float(np.mean(gate_val)),
        "gate_std_val": float(np.std(gate_val)),
        "gate_best_inner_val_mse": history["best_val_mse"],
        "gate_stopped_epoch": history["stopped_epoch"],
        "component_prediction_protocol": "5-fold GroupKFold OOF for gate training; full-train refit for outer validation",
        "random_seed": seed,
        "training_products": len(train),
        "validation_products": len(validation),
        **{f"validation_{k}": v for k, v in metrics.items()},
        "notes": (
            "MLP gate trained against five-fold training-only OOF component predictions "
            "(15% inner split for early stopping). Full-train Ridge and LightGBM are "
            "used only for outer-validation prediction; outer validation was not used during fitting."
        ),
    }
    _record(Path(results_path), row)
    return [row]
