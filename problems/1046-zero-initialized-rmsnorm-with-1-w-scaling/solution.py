import numpy as np
from numpy.typing import NDArray


def zero_init_rmsnorm(x: NDArray, weight: NDArray, eps: float = 1e-5):
    """
    RMSNorm with (1 + weight) scaling, computed in float32 and cast back.

    Args:
        x: numpy array of shape (..., D)
        weight: numpy array of shape (D,)
        eps: small constant for numerical stability

    Returns:
        A nested Python list with the same shape as x.
    """

    dtype = x.dtype
    x = x.astype(np.float32)
    rms = np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
    x_normed = x / rms
    y = x_normed * (1 + weight)
    return y.astype(dtype).tolist()
