import numpy as np
import pytest
from src.evaluation.metrics import smape

def test_smape_perfect_predictions(): assert smape([1, 2], [1, 2]) == 0
def test_smape_underprediction(): assert smape([10], [5]) == pytest.approx(66.6666667)
def test_smape_overprediction(): assert smape([5], [10]) == pytest.approx(66.6666667)
def test_smape_zero_values(): assert smape([0, 0], [0, 2]) == 100
def test_smape_batches_and_edges(): assert smape(np.array([0, 2, 4]), np.array([0, 4, 2])) == pytest.approx((0 + 66.6666667 + 66.6666667) / 3)
def test_smape_shape_error():
    with pytest.raises(ValueError): smape([1], [1, 2])
