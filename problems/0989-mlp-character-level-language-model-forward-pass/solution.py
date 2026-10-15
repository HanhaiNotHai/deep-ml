import numpy as np
from numpy.typing import NDArray


def mlp_forward_loss(
    X: NDArray, Y: NDArray, C: NDArray, W1: NDArray, b1: NDArray, W2: NDArray, b2: NDArray
):
    """
    Forward pass of an MLP character-level language model.
    Returns mean cross-entropy loss as a float.
    """

    N, k = X.shape

    emb = C[X]
    flat = emb.reshape(N, -1)
    h = np.tanh(flat @ W1 + b1)
    logits = h @ W2 + b2

    z = logits - logits.max(axis=-1, keepdims=True)
    expz = np.exp(z)
    p = expz / expz.sum(axis=-1, keepdims=True)

    return -np.mean(np.log(p[np.arange(N), Y]))
