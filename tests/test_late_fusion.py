import numpy as np
import pytest
from src.training.late_fusion import equal_weight_late_fusion

def test_equal_weight_fusion_averages_and_clips():
    assert np.allclose(equal_weight_late_fusion([2, -3], [6, 5]), [4, 2.5])

def test_equal_weight_fusion_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        equal_weight_late_fusion([1], [1, 2])
