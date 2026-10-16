import numpy as np
from numpy.typing import NDArray


def softmax(x: NDArray):
    x = np.exp(x - x.max(axis=-1, keepdims=True))
    return x / x.sum(axis=-1, keepdims=True)


def multi_head_attention_stack(
    x: list[list[list[float]]], heads: list[tuple[list[list[float]], ...]]
):
    """
    Multi-head causal self-attention by stacking single-head attention modules.

    Args:
        x: array-like of shape (B, T, d_in)
        heads: list of H tuples (W_q, W_k, W_v); each weight is shape (d_in, d_head)

    Returns:
        numpy.ndarray of shape (B, T, H * d_head)
    """

    x: NDArray = np.array(x)
    os = []
    for wq, wk, wv in heads:
        q = x @ wq
        k = x @ wk
        v = x @ wv

        s = q @ k.swapaxes(-1, -2) / np.sqrt(k.shape[-1])
        mask = np.tril(np.ones_like(s))
        s = np.where(mask, s, -np.inf)
        a = softmax(s)
        o = a @ v

        os.append(o)
    return np.concatenate(os, axis=-1)
