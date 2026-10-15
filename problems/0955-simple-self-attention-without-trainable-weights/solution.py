import numpy as np
from numpy.typing import NDArray


def softmax(x: NDArray):
    shifted = x - x.max(axis=-1, keepdims=True)
    expx = np.exp(shifted)
    return expx / expx.sum(axis=-1, keepdims=True)


def simple_self_attention(X: list[list[float]]) -> list[list[float]]:
    """
    Compute context vectors using simple self-attention (no trainable weights).

    Args:
        X: Input embeddings of shape (T, d) as a list of lists.

    Returns:
        Context vectors of shape (T, d) as a list of lists.
    """

    x = np.array(X)
    return (softmax(x @ x.T) @ x).tolist()
