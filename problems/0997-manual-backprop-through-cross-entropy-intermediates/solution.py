import numpy as np
from numpy.typing import NDArray


def manual_cross_entropy_backward(logits: NDArray, y: NDArray) -> dict:
    """
    Manually compute gradients through every intermediate of softmax + NLL loss.

    Args:
        logits: numpy array of shape (n, C)
        y:      numpy array of integer class indices, shape (n,)

    Returns:
        Dictionary with keys 'dlogprobs', 'dprobs', 'dcounts_sum_inv',
        'dcounts_sum', 'dcounts', 'dnorm_logits', 'dlogit_maxes', 'dlogits'.
    """

    n, C = logits.shape

    logit_maxes = logits.max(axis=1, keepdims=True)
    norm_logits = logits - logit_maxes
    counts = np.exp(norm_logits)
    counts_sum = counts.sum(axis=1, keepdims=True)
    counts_sum_inv = counts_sum**-1
    probs = counts * counts_sum_inv
    logprobs = np.log(probs)
    loss = -np.mean(logprobs[range(n), y])

    dlogprobs = np.zeros_like(logprobs)
    dlogprobs[range(n), y] = -1 / n
    dprobs = dlogprobs / probs
    dcounts_sum_inv = (counts * dprobs).sum(axis=1, keepdims=True)
    dcounts_sum = -(counts_sum**-2) * dcounts_sum_inv
    dcounts = counts_sum_inv * dprobs + dcounts_sum
    dnorm_logits = counts * dcounts
    dlogit_maxes = -dnorm_logits.sum(axis=1, keepdims=True)
    dlogits = dnorm_logits

    return {
        'dlogprobs': dlogprobs,
        'dprobs': dprobs,
        'dcounts_sum_inv': dcounts_sum_inv,
        'dcounts_sum': dcounts_sum,
        'dcounts': dcounts,
        'dnorm_logits': dnorm_logits,
        'dnorm_logits': dnorm_logits,
        'dlogit_maxes': dlogit_maxes,
        'dlogits': dlogits,
    }
