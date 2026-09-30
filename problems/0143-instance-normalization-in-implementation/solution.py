import torch


def instance_normalization(
    X: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, epsilon: float = 1e-5
) -> torch.Tensor:
    """
    Perform Instance Normalization over a 4D tensor X of shape (B, C, H, W).
    gamma: scale parameter of shape (C,)
    beta: shift parameter of shape (C,)
    epsilon: small value for numerical stability
    Returns: normalized tensor of same shape as X
    """

    mu = X.mean(dim=(2, 3), keepdim=True)
    var = ((X - mu) ** 2).mean(dim=(2, 3), keepdim=True)
    return (X - mu) / torch.sqrt(var + epsilon) * gamma + beta
