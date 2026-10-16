import numpy as np
from numpy.typing import NDArray


def softmax(x: NDArray):
    x = np.exp(x - x.max(axis=-1, keepdims=True))
    return x / x.sum(axis=-1, keepdims=True)


def attention(Q: NDArray, K: NDArray, V: NDArray):
    """
    Compute attention over a sequence.

    Args:
        Q: Query matrix, shape (batch_size, query_len, dim)
        K: Key matrix, shape (batch_size, key_len, dim)
        V: Value matrix, shape (batch_size, key_len, dim)

    Returns:
        output: Attended values, shape (batch_size, query_len, dim)

    The attention mechanism should:
    1. Compute compatibility between queries and keys
    2. Convert to attention weights (non-negative, sum to 1)
    3. Use weights to compute weighted sum of values
    """

    h = 8

    q, k, v = map(lambda x: x.reshape(*x.shape[:-1], h, -1), (Q, K, V))
    q, k, v = map(lambda x: x.swapaxes(-2, -3), (q, k, v))

    s = q @ k.swapaxes(-1, -2)
    mask = np.tril(np.ones_like(s))
    s = np.where(mask, s, -np.inf)
    a = softmax(s)
    o = a @ v

    o = o.swapaxes(-2, -3)
    o = o.reshape(*o.shape[:-2], -1)

    return o
