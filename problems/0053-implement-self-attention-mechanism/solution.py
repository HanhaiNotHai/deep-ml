import numpy as np
from numpy.typing import NDArray


def compute_qkv(X, W_q, W_k, W_v):
    """Compute Query, Key, Value matrices from input X and weight matrices."""
    Q = np.dot(X, W_q)
    K = np.dot(X, W_k)
    V = np.dot(X, W_v)
    return Q, K, V


def self_attention(Q: NDArray, K: NDArray, V: NDArray):
    """
    Compute scaled dot-product self-attention.

    Args:
        Q: Query matrix of shape (seq_len, d_k)
        K: Key matrix of shape (seq_len, d_k)
        V: Value matrix of shape (seq_len, d_v)

    Returns:
        Attention output of shape (seq_len, d_v)
    """
    S: NDArray = Q @ K.T / np.sqrt(Q.shape[1])
    expS = np.exp(S)
    A: NDArray = expS / expS.sum(axis=1, keepdims=True)
    return A @ V
