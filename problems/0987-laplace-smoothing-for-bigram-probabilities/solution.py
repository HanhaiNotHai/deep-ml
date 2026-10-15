import numpy as np


def smooth_bigram_probs(N: list[list[int]], k: int):
    """
    N: 2D list or array of bigram counts (V x V)
    k: float, add-k smoothing constant (k >= 0)
    Returns: 2D list of smoothed bigram probabilities (V x V).
    """

    n = np.array(N)
    n += k
    return n / n.sum(axis=1, keepdims=True)
