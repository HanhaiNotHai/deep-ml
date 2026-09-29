import numpy as np


def log_softmax(scores: list) -> np.ndarray:
    x = np.asarray(scores)
    return x - x.max() - np.log(np.exp(x - x.max()).sum())
