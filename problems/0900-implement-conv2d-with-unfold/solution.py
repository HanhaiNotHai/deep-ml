import torch
import torch.nn.functional as F
from torch import Tensor


def conv2d_via_unfold(x: Tensor, W: Tensor, b: Tensor, stride: int = 1, padding: int = 0):
    '''implement conv2d using F.unfold + matmul'''
    return (
        W.reshape(W.shape[0], -1) @ F.unfold(x, W.shape[-2:], padding=padding, stride=stride)
    ).unflatten(dim=-1, sizes=((x.shape[-2] + 2 * padding - W.shape[-2]) // stride + 1, -1)) + b[
        None, :, None, None
    ]
