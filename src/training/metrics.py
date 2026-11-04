import numpy as np


def denormalize(y: np.ndarray, mean0: float, std0: float) -> np.ndarray:
    return y * std0 + mean0


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))
