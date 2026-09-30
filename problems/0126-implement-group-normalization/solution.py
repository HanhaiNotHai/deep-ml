import torch


def group_normalization(
    X: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    num_groups: int,
    epsilon: float = 1e-5,
) -> torch.Tensor:
    """
    Perform Group Normalization on a 4D input tensor.

    Args:
        X: torch tensor of shape (B, C, H, W), input data
        gamma: torch tensor of shape (1, C, 1, 1), scale parameter
        beta: torch tensor of shape (1, C, 1, 1), shift parameter
        num_groups: number of groups for normalization
        epsilon: small constant to avoid division by zero

    Returns:
        norm_X: torch tensor of shape (B, C, H, W), normalized output
    """

    X = X.reshape(X.shape[0], num_groups, -1, *X.shape[-2:])
    mu = X.mean(dim=(2, 3, 4), keepdim=True)
    var = ((X - mu) ** 2).mean(dim=(2, 3, 4), keepdim=True)
    y = (X - mu) / torch.sqrt(var + epsilon)
    y = y.reshape(X.shape[0], -1, *X.shape[-2:])
    return y * gamma + beta
