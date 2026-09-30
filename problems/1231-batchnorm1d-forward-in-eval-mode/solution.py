import torch
from torch import Tensor


def bn_eval(x: Tensor, mean: Tensor, var: Tensor, gamma: Tensor, beta: Tensor, eps: float = 1e-5):
    """Apply batch-norm inference normalization.

    Args:
        x (Tensor): input tensor
        mean (Tensor): running mean
        var (Tensor): running variance
        gamma (Tensor): scale parameter
        beta (Tensor): shift parameter
        eps (float): numerical stability constant

    Returns:
        Tensor: normalized and affine-transformed tensor
    """

    return (x - mean) / torch.sqrt(var + eps) * gamma + beta
