import numpy as np


def ridge_loss(X: np.ndarray, w: np.ndarray, y_true: np.ndarray, alpha: float) -> float:
    return ((y_true - X @ w) ** 2).mean() + alpha * (w**2).sum()
