import numpy as np


def triu_ones(n: int) -> np.ndarray:
    """n x n upper-triangular matrix of ones (including diagonal)."""

    i = np.arange(n)
    return (i[:, None] <= i[None]).astype(np.float64)


def cumsum(a: np.ndarray) -> np.ndarray:
    """Inclusive cumulative sum of 1-D array a without np.cumsum."""

    return a @ triu_ones(len(a))
