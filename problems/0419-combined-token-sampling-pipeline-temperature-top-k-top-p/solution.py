import numpy as np
from numpy.typing import NDArray


def combined_sampling(
    logits: list[float],
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    seed: int = 42,
) -> dict:
    """
    Apply a combined sampling pipeline to raw logits.

    Args:
            logits: Raw unnormalized scores for each token
            temperature: Scaling factor for logit sharpness (0 = greedy)
            top_k: Number of top tokens to keep (0 = disabled)
            top_p: Cumulative probability threshold for nucleus sampling (1.0 = disabled)
            seed: Random seed for reproducible sampling

    Returns:
            Dict with 'probabilities' (list of floats rounded to 4 decimals)
            and 'sampled_token' (int index of chosen token)
    """

    n = len(logits)
    logits: NDArray = np.asarray(logits, dtype=np.float64)

    if temperature <= 0:
        sampled_token = int(np.argmax(logits))
        probabilities = [0.0] * n
        probabilities[sampled_token] = 1.0

        return {
            'probabilities': probabilities,
            'sampled_token': sampled_token,
        }

    z = logits / temperature

    if 0 < top_k < n:
        indices = np.argsort(z, kind='stable')[: n - top_k]
        z[indices] = -np.inf

    expz = np.exp(z - z.max())
    probs = expz / expz.sum()

    if top_p < 1.0:
        indices = np.argsort(-probs, kind='stable')
        sorted_probs = probs[indices]
        cumulative_before = np.concatenate(([0.0], np.cumsum(sorted_probs[:-1])))

        keep = indices[cumulative_before < top_p]
        filtered = np.zeros_like(probs)
        filtered[keep] = probs[keep]
        probs = filtered / filtered.sum()

    rng = np.random.default_rng(seed)
    sampled_token = int(rng.choice(n, p=probs))
    return {
        'probabilities': np.round(probs, 4).tolist(),
        'sampled_token': sampled_token,
    }
