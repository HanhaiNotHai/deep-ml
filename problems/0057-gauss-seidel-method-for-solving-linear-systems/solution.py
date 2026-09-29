import numpy as np


def gauss_seidel(A, b, n, x_ini=None):
    x = x_ini or np.zeros_like(b)
    for _ in range(n):
        for i in range(x.shape[0]):
            x[i] = (b[i] - A[i, :i] @ x[:i] - A[i, i + 1 :] @ x[i + 1 :]) / A[i, i]
    return x
