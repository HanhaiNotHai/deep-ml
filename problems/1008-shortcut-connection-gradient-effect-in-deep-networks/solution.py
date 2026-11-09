import numpy as np
from numpy.typing import NDArray


def sigmoid(z: NDArray) -> NDArray:
    return 1 / (1 + np.exp(-z))


def shortcut_gradients(
    weights: list[NDArray], biases: list[NDArray], x: NDArray, target: NDArray, use_shortcut: bool
):
    """Compute the mean absolute gradient of the weight matrix in each layer.

    Args:
        weights: list of np.ndarray weight matrices, one per layer.
        biases:  list of np.ndarray bias vectors, one per layer.
        x:       1D np.ndarray input vector.
        target:  1D np.ndarray target vector matching the network output shape.
        use_shortcut: bool, whether to add residual connections when shapes match.

    Returns:
        List of floats: mean(abs(dL/dW_i)) for each layer i, rounded to 4 dp.
    """

    L = len(weights)

    w: list[NDArray] = [None] + weights
    b: list[NDArray] = [None] + biases
    x: list[NDArray] = [x] + [None] * L
    z: list[NDArray] = [None] * (L + 1)
    a: list[NDArray] = [None] * (L + 1)
    for l in range(1, L + 1):
        z[l] = w[l] @ x[l - 1] + b[l]
        a[l] = sigmoid(z[l])
        x[l] = x[l - 1] + a[l] if use_shortcut else a[l]

    dx: list[NDArray] = [None] * L + [2 * (x[L] - target)]
    da: list[NDArray] = [None] * (L + 1)
    dz: list[NDArray] = [None] * (L + 1)
    dw: list[NDArray] = [None] * (L + 1)
    for l in reversed(range(1, L + 1)):
        da[l] = dx[l]
        dz[l] = da[l] * a[l] * (1 - a[l])
        dw[l] = np.outer(dz[l], x[l - 1])
        dx[l - 1] = w[l].T @ dz[l]
        if use_shortcut:
            dx[l - 1] += dx[l]

    return [round(np.mean(np.abs(dw[l])), 4) for l in range(1, L + 1)]
