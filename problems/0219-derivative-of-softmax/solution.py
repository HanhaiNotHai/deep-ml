from functools import partial

import torch


def softmax_derivative(x: torch.Tensor) -> torch.Tensor:
    """
    Compute the Jacobian matrix of the softmax function using PyTorch.

    Args:
        x: Input tensor

    Returns:
        Jacobian matrix J where J[i][j] = d(softmax_i)/d(x_j)
    """

    return torch.autograd.functional.jacobian(partial(torch.softmax, dim=-1), x)
