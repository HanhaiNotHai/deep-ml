import numpy as np
from numpy.typing import NDArray


def rref(matrix: NDArray):
    mat = matrix.astype(np.float64)
    m, n = mat.shape

    c = -1
    for r in range(m):
        c += 1
        while c < n and np.all(mat[r:, c] == 0):
            c += 1
        if c >= n:
            break

        for i in range(r, m):
            if mat[i, c] != 0:
                break
        if r != i:
            mat[[r, i]] = mat[[i, r]]

        mat[r] /= mat[r, c]

        other_row = np.arange(m) != r
        mat[other_row] -= mat[other_row, c : c + 1] * mat[r]

    return mat
