import numpy as np
from numpy.typing import NDArray


def batchnorm1d(
    x: NDArray,
    gamma: NDArray,
    beta: NDArray,
    running_mean: NDArray,
    running_var: NDArray,
    training: bool = True,
    momentum: float = 0.1,
    eps: float = 1e-5,
):
    """
    BatchNorm1d supporting 2D (N, C) and 3D (N, L, C) inputs.

    Returns a dict with keys 'out', 'running_mean', 'running_var'.
    """

    axis = tuple(range(x.ndim - 1))

    if training:
        mu = np.mean(x, axis)
        var = np.var(x, axis)
        x_hat = (x - mu) / np.sqrt(var + eps)
        out = gamma * x_hat + beta

        running_mean = (1 - momentum) * running_mean + momentum * mu
        running_var = (1 - momentum) * running_var + momentum * var
    else:
        x_hat = (x - running_mean) / np.sqrt(running_var + eps)
        out = gamma * x_hat + beta

    return {
        'out': out.tolist(),
        'running_mean': running_mean.tolist(),
        'running_var': running_var.tolist(),
    }
