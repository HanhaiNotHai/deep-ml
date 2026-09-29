from typing import List

import torch


def log_softmax(scores: List[float]) -> torch.Tensor:
    """
    Compute the log-softmax of a 1D list of scores using PyTorch.
    Args:
        scores: list of floats
    Returns:
        torch.Tensor of log-softmax values
    """

    x = torch.tensor(scores, dtype=torch.float32)
    max_x = x.max(dim=-1, keepdim=True).values
    return x - max_x - torch.log(torch.sum(torch.exp(x - max_x), dim=-1))
