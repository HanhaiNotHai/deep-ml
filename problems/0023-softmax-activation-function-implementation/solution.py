import torch
import torch.nn.functional as F


def softmax(scores: list[float]) -> list[float]:
    """
    Compute the softmax activation function using PyTorch's built-in API.
    Input:
      - scores: list of floats (logits)
    Returns:
      - list of floats representing the softmax probabilities.
    """

    return F.softmax(torch.tensor(scores, dtype=torch.float32), dim=-1).tolist()
