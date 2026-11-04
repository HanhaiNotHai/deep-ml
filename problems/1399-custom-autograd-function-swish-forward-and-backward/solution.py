import torch
from torch import Tensor
from torch.autograd.function import FunctionCtx


class SwishFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx: FunctionCtx, x: Tensor):
        ctx.save_for_backward(x)
        return x * torch.sigmoid(x)

    @staticmethod
    def backward(ctx: FunctionCtx, grad_output: Tensor):
        (x,) = ctx.saved_tensors
        return grad_output * torch.sigmoid(x) * (1 + x * (1 - torch.sigmoid(x)))


def swish(x: Tensor):
    return SwishFunction.apply(x)
