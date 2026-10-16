import torch


def top_p_sampling(logits: list[float], p: float) -> torch.Tensor:
    """
    Apply top-p (nucleus) sampling to filter a probability distribution.

    Args:
        logits: Raw unnormalized scores for each token
        p: Cumulative probability threshold (0 < p <= 1)

    Returns:
        Filtered and renormalized probability distribution as a torch.Tensor
    """

    logits = torch.tensor(logits, dtype=torch.float32)

    probs = torch.softmax(logits, dim=-1)

    sorted_indices = torch.argsort(probs, descending=True, stable=True)
    sorted_probs = probs[sorted_indices]

    cumulative = torch.cumsum(sorted_probs, dim=-1)
    cutoff = torch.searchsorted(
        cumulative, torch.tensor(p, dtype=torch.float32), right=False
    ).item()

    nucleus_indices = sorted_indices[: cutoff + 1]

    filtered = torch.zeros_like(probs)
    filtered[nucleus_indices] = probs[nucleus_indices]
    filtered /= filtered.sum(dim=-1, keepdim=True)

    return torch.round(filtered * 10000) / 10000
