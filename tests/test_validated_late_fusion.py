import numpy as np
import pytest
from src.training.validated_late_fusion import select_blend_weight

def test_select_blend_weight_favours_better_image_prediction():
    weight, score = select_blend_weight(np.array([10., 20.]), np.array([0., 0.]), np.array([10., 20.]), np.array([0., .5, 1.]))
    assert weight == 0.0 and score == 0.0

def test_select_blend_weight_rejects_out_of_range_weights():
    with pytest.raises(ValueError):
        select_blend_weight(np.array([1.]), np.array([1.]), np.array([1.]), np.array([-0.1]))
