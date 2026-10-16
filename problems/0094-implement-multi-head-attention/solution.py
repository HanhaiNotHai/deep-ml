import math
from typing import Tuple

import torch
import torch.nn.functional as F


def compute_qkv(
    X: torch.Tensor, W_q: torch.Tensor, W_k: torch.Tensor, W_v: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute Query, Key, and Value matrices.

    Args:
        X: Input matrix of shape (seq_len, d_model)
        W_q, W_k, W_v: Weight matrices of shape (d_model, d_model)

    Returns:
        Q, K, V matrices each of shape (seq_len, d_model)
    """

    return tuple(map(lambda W: X @ W, (W_q, W_k, W_v)))


def self_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Compute scaled dot-product self-attention.

    Args:
        Q: Query matrix of shape (seq_len, d_k)
        K: Key matrix of shape (seq_len, d_k)
        V: Value matrix of shape (seq_len, d_k)

    Returns:
        Attention output of shape (seq_len, d_k)
    """

    return F.softmax(Q @ K.transpose(-1, -2) / math.sqrt(K.shape[-1]), dim=-1) @ V


def multi_head_attention(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, n_heads: int
) -> torch.Tensor:
    """
    Compute multi-head attention.

    Args:
        Q, K, V: Matrices of shape (seq_len, d_model)
        n_heads: Number of attention heads

    Returns:
        Attention output of shape (seq_len, d_model)
    """

    Q, K, V = map(lambda x: x.reshape(*x.shape[:-1], n_heads, -1), (Q, K, V))
    Q, K, V = map(lambda x: x.transpose(-2, -3), (Q, K, V))
    o = self_attention(Q, K, V)
    o = o.transpose(-2, -3)
    o = o.reshape(*o.shape[:-2], -1)
    return o
