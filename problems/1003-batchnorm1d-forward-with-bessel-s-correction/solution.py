import numpy as np


def batchnorm1d_forward(
    X: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-5
) -> np.ndarray:
    """
    BatchNorm1d forward pass with Bessel's correction (unbiased variance).

    Args:
        X:     input batch of shape (N, D)
        gamma: scale parameter of shape (D,)
        beta:  shift parameter of shape (D,)
        eps:   numerical stability constant

    Returns:
        Output array of shape (N, D)
    """

    mean = X.mean(axis=0)
    var = ((X - mean) ** 2).sum(axis=0) / (X.shape[0] - 1)
    return gamma * (X - mean) / np.sqrt(var + eps) + beta
