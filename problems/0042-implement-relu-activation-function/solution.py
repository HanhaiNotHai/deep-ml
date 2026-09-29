import torch


def relu(z: float) -> torch.Tensor:
    """
    Implements the ReLU activation function using PyTorch.

    Args:
        z: A float input value.

    Returns:
        A torch.Tensor with ReLU applied (max(0, z)).
    """

    return torch.maximum(torch.tensor(0), torch.tensor(z))
