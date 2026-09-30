import torch


def layer_normalization(
    X: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, epsilon: float = 1e-5
) -> torch.Tensor:
    """
    Perform Layer Normalization.
    """

    mu = X.mean(dim=-1, keepdim=True)
    var = ((X - mu) ** 2).mean(dim=-1, keepdim=True)
    return (X - mu) / torch.sqrt(var + epsilon) * gamma + beta
