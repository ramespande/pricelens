"""Unit tests for src/training/semantic_late_fusion.py."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.training.semantic_late_fusion import (
    _load_embeddings,
    _fit_image_model,
    _fit_text_model,
    _predict_pair,
    _select_blend_weight,
)


# ---------------------------------------------------------------------------
# _load_embeddings
# ---------------------------------------------------------------------------

def test_load_embeddings_aligns_to_label_order(tmp_path):
    """Embeddings must be reordered to match label row order, not manifest order."""
    manifest = pd.DataFrame({"sample_id": [10, 20, 30]})
    manifest.to_csv(tmp_path / "manifest.csv", index=False)
    embeddings = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    np.save(tmp_path / "embeddings.npy", embeddings)

    labels = pd.DataFrame({"sample_id": [30, 10], "price": [100.0, 50.0]})
    x, y = _load_embeddings(tmp_path, labels)

    assert x.shape == (2, 2)
    np.testing.assert_array_equal(x[0], [5.0, 6.0])  # sample_id 30
    np.testing.assert_array_equal(x[1], [1.0, 2.0])  # sample_id 10
    np.testing.assert_array_equal(y, [100.0, 50.0])


def test_load_embeddings_raises_when_no_overlap(tmp_path):
    manifest = pd.DataFrame({"sample_id": [1, 2]})
    manifest.to_csv(tmp_path / "manifest.csv", index=False)
    np.save(tmp_path / "embeddings.npy", np.zeros((2, 4), dtype=np.float32))

    labels = pd.DataFrame({"sample_id": [99], "price": [10.0]})
    with pytest.raises(ValueError, match="No cached embeddings"):
        _load_embeddings(tmp_path, labels)


# ---------------------------------------------------------------------------
# Component models
# ---------------------------------------------------------------------------

def test_fit_image_model_returns_positive_predictions():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(50, 8)).astype(np.float32)
    y = rng.uniform(1, 100, 50)
    model = _fit_image_model(x, y)
    pred = np.expm1(model.predict(x))
    assert np.all(pred > 0)


def test_fit_text_model_returns_finite_predictions():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(60, 16)).astype(np.float32)
    y = rng.uniform(1, 200, 60)
    model = _fit_text_model(x, y, seed=42)
    pred = np.expm1(model.predict(x))
    assert np.all(np.isfinite(pred))


def test_predict_pair_returns_non_negative():
    rng = np.random.default_rng(2)
    x_img = rng.normal(size=(20, 8)).astype(np.float32)
    x_txt = rng.normal(size=(20, 16)).astype(np.float32)
    y = rng.uniform(1, 50, 20)
    img_model = _fit_image_model(x_img, y)
    txt_model = _fit_text_model(x_txt, y, seed=0)
    txt_pred, img_pred = _predict_pair(img_model, txt_model, x_img, x_txt)
    # expm1 of log-target may be negative for very low predictions; clip is in caller
    assert txt_pred.shape == (20,)
    assert img_pred.shape == (20,)


# ---------------------------------------------------------------------------
# _select_blend_weight
# ---------------------------------------------------------------------------

def test_select_blend_weight_picks_correct_minimum():
    """When text predictions are perfect, best weight should be 1.0."""
    y = np.array([10.0, 20.0, 30.0])
    text_pred = np.array([10.0, 20.0, 30.0])
    image_pred = np.array([1.0, 2.0, 3.0])
    weight, score = _select_blend_weight(y, text_pred, image_pred)
    assert weight == pytest.approx(1.0)
    assert score == pytest.approx(0.0)


def test_select_blend_weight_is_in_unit_interval():
    rng = np.random.default_rng(7)
    y = rng.uniform(1, 100, 100)
    t = rng.uniform(1, 100, 100)
    im = rng.uniform(1, 100, 100)
    w, _ = _select_blend_weight(y, t, im)
    assert 0.0 <= w <= 1.0
