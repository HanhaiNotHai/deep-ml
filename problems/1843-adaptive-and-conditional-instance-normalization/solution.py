import numpy as np
from numpy.typing import NDArray


def cin(phi_c: NDArray, gamma_s: NDArray, beta_s: NDArray, eps: float = 1e-5):
    if gamma_s.ndim == 1:
        gamma_s = gamma_s[None, :, None, None]
    if beta_s.ndim == 1:
        beta_s = beta_s[None, :, None, None]
    return (
        gamma_s
        * (phi_c - phi_c.mean(axis=(-1, -2), keepdims=True))
        / np.sqrt(phi_c.var(axis=(-1, -2), keepdims=True) + eps)
        + beta_s
    )


def adain(phi_c: NDArray, phi_s: NDArray, eps: float = 1e-5):
    return cin(
        phi_c,
        np.sqrt(phi_s.var(axis=(-1, -2), keepdims=True) + eps),
        phi_s.mean(axis=(-1, -2), keepdims=True),
    )
