import numpy as np
from numpy.typing import NDArray


def softmax(x: NDArray):
    x = np.exp(x - x.max(axis=-1, keepdims=True))
    return x / x.sum(axis=-1, keepdims=True)


def batched_causal_self_attention(
    X: list[list[list[float]]],
    W_query: list[list[float]],
    W_key: list[list[float]],
    W_value: list[list[float]],
):
    """
    Batched causal self-attention.

    Args:
        X: nested list of shape (B, T, d_in)
        W_query, W_key, W_value: nested lists of shape (d_in, d_out)

    Returns:
        Nested list of shape (B, T, d_out) -- the context vectors.
    """

    x, Wq, Wk, Wv = map(np.array, (X, W_query, W_key, W_value))

    T = x.shape[1]

    q = x @ Wq
    k = x @ Wk
    v = x @ Wv

    return (
        softmax(
            np.where(
                np.tril(np.ones((T, T))), q @ k.swapaxes(-1, -2) / np.sqrt(k.shape[-1]), -np.inf
            )
        )
        @ v
    )
