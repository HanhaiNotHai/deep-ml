import numpy as np


def mini_batch_kmeans(
    X: np.ndarray, k: int, batch_size: int, max_iters: int, seed: int = 42
) -> list:
    """
    Perform Mini-Batch K-Means clustering.

    Args:
        X: Data points of shape (n_samples, n_features)
        k: Number of clusters
        batch_size: Size of each mini-batch
        max_iters: Number of iterations
        seed: Random seed for reproducibility

    Returns:
        List of k centroids, each centroid is a list of coordinates
    """
    np.random.seed(seed)

    n_samples, n_features = X.shape
    centroids = X[np.random.choice(n_samples, k, replace=False)]
    vc = np.zeros(k)

    for _ in range(max_iters):
        batch = X[np.random.choice(n_samples, batch_size, replace=False)]

        distances_squared = np.array(
            [((batch - centroid) ** 2).sum(axis=1) for centroid in centroids]
        )
        assignments = np.argmin(distances_squared, axis=0)

        for i in range(k):
            for x in batch[assignments == i]:
                vc[i] += 1
                lr = 1 / vc[i]
                centroids[i] = (1 - lr) * centroids[i] + lr * x

    return [list(map(float, centroid)) for centroid in np.round(centroids, 4)]
