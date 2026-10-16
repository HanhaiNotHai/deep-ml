import numpy as np
from numpy.typing import NDArray


def ln(x: NDArray, gamma: NDArray, beta: NDArray, eps: float = 1e-5):
    return (
        gamma
        * (x - np.mean(x, axis=-1, keepdims=True))
        / np.sqrt(np.var(x, axis=-1, keepdims=True) + eps)
        + beta
    )


def softmax(x: NDArray):
    x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return x / np.sum(x, axis=-1, keepdims=True)


def mha(x: NDArray, W_q: NDArray, W_k: NDArray, W_v: NDArray, W_o: NDArray, num_heads: int):
    qkv: map[NDArray] = map(lambda W: x @ W, (W_q, W_k, W_v))
    qkv = map(lambda x: x.reshape(*x.shape[:-1], num_heads, -1), qkv)
    q, k, v = map(lambda x: x.swapaxes(-2, -3), qkv)

    s = q @ k.swapaxes(-1, -2) / np.sqrt(k.shape[-1])
    mask = np.tril(np.ones_like(s))
    s = np.where(mask, s, -np.inf)
    a = softmax(s)
    o = a @ v

    o = o.swapaxes(-2, -3)
    o = o.reshape(*o.shape[:-2], -1)
    return o @ W_o


def gelu(x: NDArray) -> NDArray:
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x * x * x)))


def ffn(x: NDArray, W_ff1: NDArray, b_ff1: NDArray, W_ff2: NDArray, b_ff2: NDArray):
    return gelu(x @ W_ff1 + b_ff1) @ W_ff2 + b_ff2


def pre_norm_transformer_block(x: NDArray, params: dict[str, NDArray], num_heads: int):
    """
    Pre-norm Transformer block forward pass.

    Args:
        x: numpy array of shape (batch, seq_len, emb_dim)
        params: dict with keys 'ln1_gamma','ln1_beta','ln2_gamma','ln2_beta',
                'W_q','W_k','W_v','W_o','W_ff1','b_ff1','W_ff2','b_ff2'
        num_heads: int, number of attention heads (emb_dim must be divisible by num_heads)

    Returns:
        numpy array of shape (batch, seq_len, emb_dim)
    """

    x += mha(
        ln(x, params['ln1_gamma'], params['ln1_beta']),
        params['W_q'],
        params['W_k'],
        params['W_v'],
        params['W_o'],
        num_heads,
    )
    x += ffn(
        ln(x, params['ln2_gamma'], params['ln2_beta']),
        params['W_ff1'],
        params['b_ff1'],
        params['W_ff2'],
        params['b_ff2'],
    )
    return x
