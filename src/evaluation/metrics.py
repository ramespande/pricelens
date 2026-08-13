"""Competition-aligned metrics evaluated on the original price scale."""
from __future__ import annotations
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

def smape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Symmetric mean absolute percentage error in percent, safely handling 0/0."""
    actual_array = np.asarray(actual, dtype=float)
    predicted_array = np.asarray(predicted, dtype=float)
    if actual_array.shape != predicted_array.shape:
        raise ValueError("actual and predicted must have the same shape")
    denominator = (np.abs(actual_array) + np.abs(predicted_array)) / 2.0
    ratios = np.divide(np.abs(predicted_array - actual_array), denominator, out=np.zeros_like(denominator), where=denominator != 0)
    return float(np.mean(ratios) * 100)

def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    predicted = np.clip(np.asarray(predicted, dtype=float), 0, None)
    return {"smape": smape(actual, predicted), "mae": float(mean_absolute_error(actual, predicted)), "rmse": float(mean_squared_error(actual, predicted) ** 0.5)}
