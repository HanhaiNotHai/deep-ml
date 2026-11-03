import numpy as np
from numpy.typing import NDArray


def Swish(z: NDArray) -> NDArray:
    return z / (1 + np.exp(-z))


def SwiGLU(x: NDArray):
    """
    Args:
        x: np.ndarray of shape (batch_size, 2d)
    Returns:
        np.ndarray of shape (batch_size, d)
    """

    x1, x2 = np.split(x, 2, -1)
    return x1 * Swish(x2)
