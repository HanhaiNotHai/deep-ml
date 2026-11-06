import numpy as np
from numpy.typing import NDArray


def cohere_layernorm(x: NDArray, weight: NDArray, eps: float = 1e-5):
    """Bias-less LayerNorm computed in float32.

    Args:
        x: numpy array of shape (..., D)
        weight: numpy array of shape (D,)
        eps: small constant for numerical stability
    Returns:
        numpy array of same shape and dtype as x
    """

    return (
        (x.astype(np.float32) - x.mean(axis=-1, keepdims=True))
        / np.sqrt(x.var(axis=-1, keepdims=True) + eps)
        * weight
    ).astype(x.dtype)
