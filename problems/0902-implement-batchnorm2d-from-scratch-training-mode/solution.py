import torch
from torch import Tensor


def batchnorm2d(x: Tensor, gamma: Tensor, beta: Tensor, eps: float = 1e-5):
    '''training-mode batchnorm2d'''

    mu = x.mean(dim=(0, -2, -1), keepdim=True)
    var = ((x - mu) ** 2).mean(dim=(0, -2, -1), keepdim=True)
    return (x - mu) / torch.sqrt(var + eps) * gamma[:, None, None] + beta[:, None, None]
