import numpy as np
from numpy.typing import NDArray


def softmax(x: NDArray):
    x = np.exp(x - x.max(axis=-1, keepdims=True))
    return x / x.sum(axis=-1, keepdims=True)


def mha_combined_qkv(x: NDArray, W_qkv: NDArray, W_out: NDArray, num_heads: int):
    """
    Multi-head causal self-attention using a single combined QKV weight matrix.

    Args:
        x: array of shape (batch, num_tokens, d_in)
        W_qkv: array of shape (d_in, 3 * d_out)
        W_out: array of shape (d_out, d_out)
        num_heads: int, number of attention heads (d_out % num_heads == 0)

    Returns:
        Nested list of shape (batch, num_tokens, d_out).
    """

    qkv = np.split(x @ W_qkv, 3, axis=-1)
    qkv = map(lambda x: x.reshape(*x.shape[:-1], num_heads, -1), qkv)
    q, k, v = map(lambda x: x.swapaxes(-2, -3), qkv)

    s = q @ k.swapaxes(-1, -2) / np.sqrt(k.shape[-2])
    mask = np.tril(np.ones_like(s))
    s = np.where(mask, s, -np.inf)
    a = softmax(s)
    o = a @ v

    o = o.swapaxes(-2, -3)
    o = o.reshape(*o.shape[:-2], -1)
    o = o @ W_out

    return o
