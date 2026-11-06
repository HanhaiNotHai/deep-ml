import numpy as np


def local_response_normalization(
    x: np.ndarray,
    n: int = 5,
    k: float = 2.0,
    alpha: float = 1e-4,
    beta: float = 0.75,
) -> np.ndarray:
    """
    Applies Local Response Normalization across the channel dimension.

    Args:
        x: Input tensor of shape (N, C, H, W).
        n: Local window size.
        k: Additive constant.
        alpha: Scaling parameter.
        beta: Exponent parameter.

    Returns:
        Normalized tensor of the same shape as input, rounded to 4 decimals.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 4:
        raise ValueError("Input must have shape (N, C, H, W).")
    if n < 1:
        raise ValueError("n must be a positive integer.")

    h = n // 2

    # Cumulative sum of squared activations along the channel axis.
    sq_sum = np.pad(
        np.cumsum(x**2, axis=1),
        ((0, 0), (1, 0), (0, 0), (0, 0)),
    )

    # Compute clipped channel window boundaries.
    channels = np.arange(x.shape[1])
    left = np.maximum(channels - h, 0)
    right = np.minimum(channels + h + 1, x.shape[1])

    # Sum of squared activations within each local window.
    local_sum = sq_sum[:, right] - sq_sum[:, left]

    # Apply Local Response Normalization.
    return np.round(x / (k + alpha * local_sum) ** beta, 4)