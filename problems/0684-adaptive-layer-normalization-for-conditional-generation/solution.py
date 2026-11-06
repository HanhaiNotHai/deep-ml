import numpy as np
from numpy.typing import NDArray


def adaptive_layer_norm(
    x: NDArray,
    c: NDArray,
    W_scale: NDArray,
    b_scale: NDArray,
    W_shift: NDArray,
    b_shift: NDArray,
    epsilon: float = 1e-5,
):
    """
    Apply Adaptive Layer Normalization.

    Args:
        x: Input features of shape (B, D)
        c: Conditioning embeddings of shape (B, C)
        W_scale: Weight matrix for scale projection, shape (C, D)
        b_scale: Bias for scale projection, shape (D,)
        W_shift: Weight matrix for shift projection, shape (C, D)
        b_shift: Bias for shift projection, shape (D,)
        epsilon: Small constant for numerical stability

    Returns:
        Adaptively normalized output of shape (B, D)
    """

    scale = c @ W_scale + b_scale
    shift = c @ W_shift + b_shift
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    x_norm = (x - mu) / np.sqrt(var + epsilon)
    return x_norm * scale + shift
