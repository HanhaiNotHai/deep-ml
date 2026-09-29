import numpy as np


def kmeans_plus_plus_init(X: np.ndarray, k: int, seed: int = None) -> np.ndarray:
    """
    Initialize k centroids using the K-Means++ algorithm.

    Args:
            X: Data points of shape (n_samples, n_features)
            k: Number of centroids to initialize
            seed: Random seed for reproducibility

    Returns:
            Centroids of shape (k, n_features)
    """
    np.random.seed(seed)

    n_samples, n_features = X.shape
    centroids = np.empty([k, n_features])
    centroids[0] = X[np.random.randint(0, n_samples)]

    for i in range(1, k):
        distances = np.array([((X - centroid) ** 2).sum(axis=1) for centroid in centroids[:i]])
        p = distances.min(axis=0)
        p /= p.sum()
        centroids[i] = X[np.random.choice(n_samples, p=p)]

    return centroids
