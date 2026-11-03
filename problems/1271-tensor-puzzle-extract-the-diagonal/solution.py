import numpy as np


def diag(A: np.ndarray) -> np.ndarray:
    """Return the main diagonal of square matrix A."""

    i = np.arange(len(A))
    return A[i, i]
