import numpy as np


def elbow_wcss(X: np.ndarray, k_values: list, max_iters: int = 100) -> list:
    """
    Compute WCSS (inertia) for each k in k_values using K-Means.

    Args:
        X: Data of shape (n_samples, n_features)
        k_values: List of cluster counts to evaluate
        max_iters: Maximum number of Lloyd iterations

    Returns:
        List of WCSS values (rounded to 4 decimals), one per k
    """
    n_samples, n_feature = X.shape
    wcss = []
    for k in k_values:
        centroids = X[:k]

        for _ in range(max_iters):
            distances_squared = np.array(
                [((X - centroid) ** 2).sum(axis=1) for centroid in centroids]
            )
            assignments = np.argmin(distances_squared, axis=0)

            new_centroids = np.array(
                [
                    (
                        cluster.mean(axis=0)
                        if (cluster := X[assignments == i]).shape[0] > 0
                        else centroid
                    )
                    for i, centroid in enumerate(centroids)
                ]
            )

            if np.all(new_centroids == centroids):
                break
            centroids = new_centroids

        wcss.append(float(distances_squared[assignments, range(n_samples)].sum()))
    return wcss
