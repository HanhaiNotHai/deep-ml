from typing import Callable

import torch
import torch.nn as nn
from torch import Tensor


def train_step(
    model: nn.Module,
    x: Tensor,
    y: Tensor,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable[[Tensor, Tensor], Tensor],
):
    """Run one training step and return the pre-update loss as a float.

    Args:
        model: torch.nn.Module to train.
        x: Input batch tensor.
        y: Target batch tensor.
        optimizer: torch.optim optimizer bound to model parameters.
        loss_fn: Callable (pred, y) -> scalar loss tensor.

    Returns:
        float: Loss value computed before optimizer.step().
    """

    optimizer.zero_grad()
    loss = loss_fn(model(x), y)
    loss.backward()
    optimizer.step()
    return loss.item()
