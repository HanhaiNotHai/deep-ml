import numpy as np


def eye(n: int) -> np.ndarray:
    """n x n identity matrix without np.eye."""

    i = np.arange(n)
    return (i[:, None] == i[None]).astype(np.float64)
