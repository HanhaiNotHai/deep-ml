def matrix_determinant_and_trace(matrix: list[list[float]]) -> tuple[float, float]:
    """
    Compute the determinant and trace of a square matrix.

    Args:
            matrix: A square matrix (n x n) represented as list of lists

    Returns:
            Tuple of (determinant, trace)
    """

    def mat_det(mat: list[list[float]]) -> float:
        n = len(mat)
        if n == 1:
            return mat[0][0]

        det = 0
        for col in range(n):
            M = [row[:col] + row[col + 1 :] for row in mat[1:]]
            det += ((-1) ** col) * mat[0][col] * mat_det(M)

        return det

    return mat_det(matrix), sum(row[i] for i, row in enumerate(matrix))
