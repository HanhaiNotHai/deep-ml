import torch


def batch_normalization(
    X: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, epsilon: float = 1e-5
) -> torch.Tensor:
    """Perform Batch Normalization on a 4D tensor in BCHW format."""

    mu = X.mean(dim=(0, 2, 3), keepdim=True)
    var = ((X - mu) ** 2).mean(dim=(0, 2, 3), keepdim=True)
    return (X - mu) / torch.sqrt(var + epsilon) * gamma + beta
