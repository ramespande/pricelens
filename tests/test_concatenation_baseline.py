"""Unit tests for src/training/concatenation_baseline.py."""
from __future__ import annotations

import numpy as np
import pytest

from src.training.concatenation_baseline import concatenate_embeddings


def test_concatenate_correct_shape():
    text = np.ones((10, 384), dtype=np.float32)
    image = np.ones((10, 512), dtype=np.float32)
    joint = concatenate_embeddings(text, image)
    assert joint.shape == (10, 896)


def test_concatenate_preserves_values():
    text = np.full((3, 2), 1.0, dtype=np.float32)
    image = np.full((3, 3), 2.0, dtype=np.float32)
    joint = concatenate_embeddings(text, image)
    np.testing.assert_array_equal(joint[:, :2], 1.0)
    np.testing.assert_array_equal(joint[:, 2:], 2.0)


def test_concatenate_raises_on_row_mismatch():
    text = np.zeros((5, 4))
    image = np.zeros((6, 4))
    with pytest.raises(ValueError, match="must match"):
        concatenate_embeddings(text, image)


def test_concatenate_single_row():
    text = np.array([[1.0, 2.0]])
    image = np.array([[3.0, 4.0, 5.0]])
    joint = concatenate_embeddings(text, image)
    np.testing.assert_array_equal(joint, [[1.0, 2.0, 3.0, 4.0, 5.0]])
