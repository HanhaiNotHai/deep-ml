import numpy as np
from numpy.typing import NDArray


def softmax(x: NDArray):
    x -= x.max(axis=-1, keepdims=True)
    x = np.exp(x)
    return x / x.sum(axis=-1, keepdims=True)


def train_bigram_nn(
    xs: list[int],
    ys: list[int],
    vocab_size: int,
    lr: float = 10.0,
    num_iters: int = 100,
    alpha: float = 0.0,
):
    """
    Train a one-layer neural-net bigram language model with full-batch gradient descent.

    Args:
        xs: list[int] of length N - input character indices in [0, vocab_size)
        ys: list[int] of length N - target next-character indices in [0, vocab_size)
        vocab_size: int V - number of distinct characters
        lr: float - learning rate
        num_iters: int - number of gradient descent steps
        alpha: float - L2 regularization strength on mean(W**2)

    Returns:
        list[list[float]] - trained weight matrix W of shape (V, V)
    """

    n = len(xs)

    x = np.eye(vocab_size)[xs]
    w = np.zeros((vocab_size, vocab_size))
    for _ in range(num_iters):
        z = x @ w
        p = softmax(z)

        dz = p
        dz[np.arange(n), ys] -= 1
        dz /= n
        dw = x.T @ dz
        dw += 2 * alpha * w / vocab_size / vocab_size
        w -= lr * dw

    return w.tolist()
