from itertools import pairwise

import numpy as np


def he_initialize(layer_dims: list[int], method: str = 'normal', seed: int = 42) -> list:
    """
    Initialize weight matrices for a neural network using He initialization.

    Args:
        layer_dims: List of integers representing neurons per layer
        method: 'normal' or 'uniform' sampling distribution
        seed: Random seed for reproducibility

    Returns:
        List of numpy arrays, one weight matrix per adjacent layer pair
    """

    np.random.seed(seed)

    ws = []
    for i, o in pairwise(layer_dims):
        if method == 'normal':
            ws.append(np.random.randn(i, o) * np.sqrt(2 / i))
        elif method == 'uniform':
            l = np.sqrt(6 / i)
            ws.append(np.random.uniform(-l, l, (i, o)))

    return ws
