import numpy as np
from numpy.typing import NDArray


def softmax(x: NDArray):
    x = np.exp(x - x.max(axis=-1, keepdims=True))
    return x / x.sum(axis=-1, keepdims=True)


def causal_mask_attention(attn_weights: NDArray, method: str = 'tril') -> list:
    """Apply causal masking two ways and return the resulting attention matrix as a nested list."""

    if method == 'tril':
        attn_weights = np.tril(attn_weights)
        attn_weights /= attn_weights.sum(axis=-1, keepdims=True)
    elif method == 'triu':
        attn_weights = np.log(attn_weights)
        mask = np.triu(np.ones_like(attn_weights), k=1)
        attn_weights = np.where(mask, -np.inf, attn_weights)
        attn_weights = softmax(attn_weights)

    return attn_weights.tolist()
