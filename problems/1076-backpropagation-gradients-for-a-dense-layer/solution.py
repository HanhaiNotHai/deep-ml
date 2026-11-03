import numpy as np
from numpy.typing import NDArray


def sigmoid(z: NDArray) -> NDArray:
    return 1 / (1 + np.exp(-z))


def dense_backward(a_prev: NDArray, W: NDArray, b: NDArray, y: NDArray) -> dict:
    """
    Compute gradients of the squared-error cost w.r.t. W, b, and a_prev
    for a dense layer z = W @ a_prev + b followed by sigmoid activation.

    Args:
        a_prev: activations from previous layer, shape (K,)
        W: weight matrix, shape (J, K)
        b: bias vector, shape (J,)
        y: target vector, shape (J,)

    Returns:
        dict with keys 'dW' (J x K nested list), 'db' (list of length J),
        and 'da_prev' (list of length K).
    """

    z = W @ a_prev + b
    a = sigmoid(z)

    da = 2 * (a - y)
    dz = da * a * (1 - a)
    dW = np.outer(dz, a_prev)
    db = dz
    da_prev = W.T @ dz

    return {
        'dW': dW,
        'db': db,
        'da_prev': da_prev,
    }
