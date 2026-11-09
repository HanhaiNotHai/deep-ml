import numpy as np
from numpy.typing import NDArray


def information_bottleneck_loss(task_loss: float, mu: NDArray, log_var: NDArray, beta: float):
    """
    Compute the Information Bottleneck regularized loss.

    Args:
        task_loss: Precomputed task loss (scalar)
        mu: Encoder means, shape (batch_size, latent_dim)
        log_var: Encoder log-variances, shape (batch_size, latent_dim)
        beta: Trade-off parameter for compression regularization

    Returns:
        Tuple of (total_loss, mean_kl_divergence), both rounded to 4 decimal places
    """

    kl_divergence = -0.5 * np.sum(1 + log_var - mu * mu - np.exp(log_var), axis=-1)
    mean_kl_divergence = np.mean(kl_divergence)
    total_loss = task_loss + beta * mean_kl_divergence
    return total_loss, mean_kl_divergence
