import numpy as np
from numpy.typing import NDArray


def gemma_rmsnorm(x: NDArray, eps: float = 1e-6, scale=None):
    """
    Gemma-style RMSNorm along the last axis.

    Args:
        x: numpy array of shape (..., D)
        eps: small constant for numerical stability
        scale: optional 1D numpy array of shape (D,). If provided, multiply
               the normalized output by (1 + scale).

    Returns:
        numpy array of the same shape as x.
    """

    return (x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)) * (
        1 + (scale if scale is not None else 0)
    )
