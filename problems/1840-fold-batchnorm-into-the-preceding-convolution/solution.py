import numpy as np
from numpy.typing import NDArray


def fold_bn(
    W: NDArray,
    b: NDArray,
    gamma: NDArray,
    beta: NDArray,
    running_mean: NDArray,
    running_var: NDArray,
    eps: float = 1e-5,
):
    scale = gamma / np.sqrt(running_var + eps)
    W_fold = W * scale[:, None, None, None]
    b_fold = scale * ((b if b is not None else 0) - running_mean) + beta
    return W_fold, b_fold
