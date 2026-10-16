import torch
import torch.nn.functional as F


def temperature_sampling(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    Compute temperature-scaled softmax probabilities from logits.

    Args:
        logits: 1D torch.Tensor of raw model output scores
        temperature: float controlling distribution sharpness

    Returns:
        torch.Tensor of probabilities after temperature scaling
    """

    if temperature <= 0:
        probs = torch.zeros_like(logits)
        probs[logits.argmax(dim=-1)] = 1
        return probs

    return F.softmax((logits - logits.max(dim=-1, keepdim=True).values) / temperature, dim=-1)
