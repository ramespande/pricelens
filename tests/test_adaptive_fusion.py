"""Unit tests for src/training/adaptive_fusion.py."""
from __future__ import annotations

import numpy as np
import pytest

from src.training.adaptive_fusion import _AdaptiveGate


# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------

def test_gate_output_in_unit_interval():
    """Gate values must all be in (0, 1) by the sigmoid activation."""
    rng = np.random.default_rng(0)
    gate = _AdaptiveGate(d_in=16, hidden=8, seed=0)
    x = rng.normal(size=(50, 16)).astype(np.float32)
    g = gate.predict_gate(x)
    assert g.shape == (50,)
    assert np.all(g > 0) and np.all(g < 1)


def test_gate_output_shape():
    gate = _AdaptiveGate(d_in=32, hidden=16, seed=1)
    x = np.zeros((100, 32), dtype=np.float32)
    g = gate.predict_gate(x)
    assert g.shape == (100,)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def test_gate_fits_trivial_case():
    """When text predictions are always correct, the gate should converge toward g≈1."""
    rng = np.random.default_rng(42)
    n = 200
    d = 16
    x = rng.normal(size=(n, d)).astype(np.float32)
    y = rng.uniform(1, 50, n)
    log_y = np.log1p(y)

    # Text model is perfect; image model is noisy.
    log_text = log_y.copy()
    log_image = log_y + rng.normal(0, 1.0, n)

    gate = _AdaptiveGate(d_in=d, hidden=32, seed=7)
    history = gate.fit(
        x, log_text, log_image, log_y,
        epochs=100, batch_size=64, lr=1e-2, seed=7, val_fraction=0.2,
    )
    assert "best_val_mse" in history
    assert "stopped_epoch" in history

    # After fitting, gate mean should shift toward favouring text (g > 0.5).
    g = gate.predict_gate(x)
    assert float(np.mean(g)) > 0.5, f"Expected gate mean > 0.5; got {np.mean(g):.3f}"


def test_gate_best_val_mse_is_finite():
    rng = np.random.default_rng(3)
    n, d = 80, 8
    x = rng.normal(size=(n, d)).astype(np.float32)
    y = rng.uniform(5, 30, n)
    log_y = np.log1p(y)
    log_text = log_y + rng.normal(0, 0.2, n)
    log_image = log_y + rng.normal(0, 0.5, n)

    gate = _AdaptiveGate(d_in=d, hidden=16, seed=5)
    history = gate.fit(x, log_text, log_image, log_y, epochs=50, batch_size=32, seed=5)
    assert np.isfinite(history["best_val_mse"])


def test_gate_early_stopping_fires():
    """With patience=3 and constant loss, early stopping should trigger quickly."""
    rng = np.random.default_rng(9)
    n, d = 60, 4
    x = rng.normal(size=(n, d)).astype(np.float32)
    y = np.ones(n) * 10.0
    log_y = np.log1p(y)
    # Both predictions identical → loss will stagnate immediately.
    log_text = log_y.copy()
    log_image = log_y.copy()

    gate = _AdaptiveGate(d_in=d, hidden=8, seed=11)
    history = gate.fit(
        x, log_text, log_image, log_y,
        epochs=200, batch_size=32, lr=1e-3, patience=3, seed=11, val_fraction=0.2,
    )
    # With patience=3 and no improvement, should stop well before 200 epochs.
    assert history["stopped_epoch"] < 100
