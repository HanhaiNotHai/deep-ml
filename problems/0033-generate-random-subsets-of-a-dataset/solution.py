import numpy as np


def get_random_subsets(X, y, n_subsets, replacements=True):
    n = X.shape[0]
    subset_size = n if replacements else n // 2
    return [
        (X[idx := np.random.choice(n, subset_size, replace=replacements)], y[idx])
        for _ in range(n_subsets)
    ]
