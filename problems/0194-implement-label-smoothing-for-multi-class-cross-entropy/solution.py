import numpy as np
from numpy.typing import NDArray


def smooth_labels(y_true: NDArray, num_classes: int, epsilon: float):
    """
    Create smoothed one-hot target vectors.

    Args:
        y_true: Iterable[int] of shape (N,) with values in [0, K-1]
        num_classes: int, total number of classes (K)
        epsilon: float in [0, 1]

    Returns:
        np.ndarray of shape (N, K) with smoothed probabilities.
    """

    if not 0 <= epsilon <= 1:
        raise ValueError

    N = y_true.shape[0]
    K = num_classes

    y_smooth = np.full((N, K), epsilon / K)
    y_smooth[range(N), y_true] = 1 - epsilon + epsilon / K

    return y_smooth


def label_smoothing_cross_entropy(
    logits: NDArray,
    y_true: NDArray,
    num_classes: int,
    epsilon: float = 0.1,
    round_decimals: int | None = None,
):
    """
    Compute mean cross-entropy between logits and smoothed targets using stable log-softmax.

    Args:
        logits: Array-like of shape (N, K), model output scores.
        y_true: Array-like of shape (N,), integer class indices.
        num_classes: int, number of classes (K).
        epsilon: float in [0, 1].
        round_decimals: int | None, round the loss to this many decimals if given.

    Returns:
        float: Mean cross-entropy loss.
    """

    y_smooth = smooth_labels(y_true, num_classes, epsilon)
    m = np.max(logits, axis=-1, keepdims=True)
    log_softmax = logits - (m + np.log(np.sum(np.exp(logits - m), axis=-1, keepdims=True)))
    l = -np.sum(y_smooth * (log_softmax), axis=-1)
    l = np.mean(l)
    if round_decimals is not None:
        l = round(l, round_decimals)
    return l
