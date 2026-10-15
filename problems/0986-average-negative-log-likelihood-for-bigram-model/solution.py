from itertools import pairwise

import numpy as np


def bigram_avg_nll(P: np.ndarray, words: list[str], stoi: dict[str, int]):
    """Average negative log-likelihood of words under a bigram model."""

    p = [-np.log(P[stoi[c1], stoi[c2]]) for w in words for c1, c2 in pairwise('.' + w + '.')]
    return sum(p) / len(p) if p else 0.0
