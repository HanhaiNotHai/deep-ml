import numpy as np


def local_response_normalization(
    x: np.ndarray, n: int = 5, k: float = 2.0, alpha: float = 1e-4, beta: float = 0.75
) -> np.ndarray:
    """
    Applies Local Response Normalization across the channel dimension.

    Args:
        x: Input tensor of shape (N, C, H, W)
        n: Local window size
        k: Additive constant
        alpha: Scaling parameter
        beta: Exponent parameter

    Returns:
        Normalized tensor of same shape as input.
    """

    N, C, H, W = x.shape
    y = np.empty_like(x)
    for i in range(C):
        y[:, i] = (
            x[:, i]
            / (k + alpha * (x[:, max(0, i - n // 2) : min(C, i + n // 2 + 1)] ** 2).sum(axis=1))
            ** beta
        )
    return y
