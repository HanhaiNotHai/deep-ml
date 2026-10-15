from itertools import pairwise

import numpy as np


def bigram_counts(words: list[str]):
    vocab = ['.'] + sorted(set(c for w in words for c in w))
    stoi = {s: i for i, s in enumerate(vocab)}
    V = len(vocab)
    N = [[0] * V for _ in range(V)]
    for w in words:
        for a, b in pairwise('.' + w + '.'):
            N[stoi[a]][stoi[b]] += 1
    return N
