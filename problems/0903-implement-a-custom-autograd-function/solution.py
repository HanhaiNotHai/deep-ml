import torch
from torch import Tensor
from torch.autograd.function import FunctionCtx


class MyReLU(torch.autograd.Function):

    @staticmethod
    def forward(ctx: FunctionCtx, x: Tensor):
        '''save anything backward will need, then return the ReLU output'''

        ctx.save_for_backward(x)
        return torch.clamp(x, min=0)

    @staticmethod
    def backward(ctx: FunctionCtx, grad_output: Tensor):
        '''return dL/dx using the saved tensors and grad_output'''

        (x,) = ctx.saved_tensors
        return grad_output * (x > 0)
