import numpy as np
from numpy.typing import NDArray


def softmax(x: NDArray):
    shifted = x - x.max(axis=-1, keepdims=True)
    expx = np.exp(shifted)
    return expx / expx.sum(axis=-1, keepdims=True)


def scaled_attention_weights(Q: np.ndarray, K: np.ndarray) -> list:
    """
    Compute scaled dot-product attention weights.

    Args:
        Q: (n_q, d_k) query matrix
        K: (n_k, d_k) key matrix

    Returns:
        Attention weights of shape (n_q, n_k) as a nested list,
        each entry rounded to 4 decimal places.
    """

    return softmax(Q @ K.T / np.sqrt(K.shape[-1]))
