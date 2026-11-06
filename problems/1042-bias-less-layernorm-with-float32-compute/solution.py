import numpy as np
from numpy.typing import NDArray


def biasless_layernorm(x: NDArray, gamma: NDArray, eps: float = 1e-5):
    """
    Bias-less LayerNorm over the last dimension with float32 compute.

    Args:
        x: numpy array of shape (..., D)
        gamma: numpy array of shape (D,)
        eps: float, numerical stability constant

    Returns:
        numpy array of same shape and dtype as x
    """

    return (
        (x.astype(np.float32) - x.mean(axis=-1, keepdims=True))
        / np.sqrt(x.var(axis=-1, keepdims=True) + eps)
        * gamma
    ).astype(x.dtype)
