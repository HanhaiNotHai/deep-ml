import numpy as np


def matrix_rank(A: np.ndarray, tol: float = 1e-10) -> int:
    """
    Compute the rank of a matrix.

    Args:
        A: Input matrix of shape (m, n)
        tol: Tolerance for considering values as zero

    Returns:
        The rank of the matrix (integer)
    """

    def zero(x):
        return np.abs(x) < tol

    A = A.astype(float)
    m, n = A.shape

    rank = 0
    r = 0
    for c in range(n):
        i = r + np.argmax(A[r:, c])
        if zero(A[i, c]):
            continue

        A[[r, i]] = A[[i, r]]
        A[r + 1 :] -= A[r + 1 :, c : c + 1] / A[r, c] * A[r]
        rank += 1

        r += 1
        if r >= m:
            break

    return rank


def is_linearly_independent(vectors: list[list[float]]) -> bool:
    """
    Check if a set of vectors is linearly independent.

    Args:
        vectors: List of vectors, where each vector is a list of floats.
                 All vectors must have the same dimension.

    Returns:
        True if vectors are linearly independent, False otherwise.
    """

    return not vectors or matrix_rank(np.array(vectors)) == len(vectors)
