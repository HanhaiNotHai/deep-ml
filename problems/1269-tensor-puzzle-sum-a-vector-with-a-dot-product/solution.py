import numpy as np


def ones(n: int) -> np.ndarray:
    """Return a length-n float vector of ones without calling np.ones."""

    return np.arange(n, dtype=np.float64) * 0 + 1


def vector_sum(a: np.ndarray):
    """Sum elements of 1-D array a without np.sum / loops."""

    return a @ ones(len(a))
