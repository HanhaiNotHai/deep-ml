import numpy as np


def segment_image(image, k: int, max_iters: int = 100):
    """
    Segment an RGB image using K-Means clustering on pixel colors.

    Args:
        image: nested list / array of shape (H, W, 3) with pixel values.
        k: number of clusters.
        max_iters: maximum number of K-Means iterations.

    Returns:
        Segmented image as a nested list of shape (H, W, 3).
    """

    image = np.array(image, dtype=np.float64)
    H, W, D = image.shape
    image.resize(H * W, D)
    centroids = image[:k]

    for _ in range(max_iters):
        distances_squared = np.array(
            [((image - centroid) ** 2).sum(axis=1) for centroid in centroids]
        )
        assignments = distances_squared.argmin(axis=0)

        new_centroids = np.array(
            [
                (
                    cluster.mean(axis=0)
                    if (cluster := image[assignments == i]).shape[0] > 0
                    else centroid
                )
                for i, centroid in enumerate(centroids)
            ]
        )

        if np.all(new_centroids == centroids):
            break
        centroids = new_centroids

    centroids = centroids.round(4)
    for i, centroid in enumerate(centroids):
        image[assignments == i] = centroid
    image.resize(H, W, D)
    return image.tolist()
