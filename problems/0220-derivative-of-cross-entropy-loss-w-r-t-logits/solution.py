import torch
import torch.nn.functional as F


def cross_entropy_derivative(logits: torch.Tensor, target: int) -> torch.Tensor:
    """
    Compute the derivative of cross-entropy loss with respect to logits.

    Args:
        logits: Raw model outputs tensor
        target: Index of the true class

    Returns:
        Gradient tensor
    """

    probs = F.softmax(logits, dim=-1)

    one_hot = torch.zeros_like(probs)
    one_hot[target] = 1

    return probs - one_hot
