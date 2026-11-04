import numpy as np


def batchnorm1d_backward_fused(
    x: np.ndarray, bngain: np.ndarray, dhpreact: np.ndarray, eps: float = 1e-5
) -> np.ndarray:
    """
    Compute the fused backward pass through a BatchNorm1d layer.

    Args:
        x:        pre-BN input, shape (N, D)
        bngain:   scale parameter, shape (1, D)
        dhpreact: upstream gradient, shape (N, D)
        eps:      numerical stability constant

    Returns:
        dhprebn: gradient w.r.t. x, shape (N, D)
    """

    N, D = x.shape
    mu = x.mean(axis=0)
    var = x.var(axis=0, ddof=1)
    var_inv = 1 / np.sqrt(var + eps)
    x_hat = (x - mu) * var_inv
    return (
        bngain
        * var_inv
        / N
        * (
            N * dhpreact
            - dhpreact.sum(axis=0)
            - N / (N - 1) * x_hat * (dhpreact * x_hat).sum(axis=0)
        )
    )
