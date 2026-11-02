import numpy as np


def pairwise_cosine_similarity(X: list[list[int]]):
    X = np.asarray(X, dtype=np.float64)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X_norm = X / np.where(norms == 0, 1.0, norms)
    sim_mat = X_norm @ X_norm.T
    return np.round(sim_mat, 4).tolist()
