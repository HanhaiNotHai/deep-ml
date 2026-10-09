from typing import Callable, Iterable

import torch
import torch.nn as nn
from torch import Tensor


def accumulated_step(
    model: nn.Module,
    micro_batches: Iterable[tuple[Tensor, Tensor]],
    optimizer: torch.optim.Optimizer,
    criterion: Callable[[Tensor, Tensor], Tensor],
):
    '''zero grads, accumulate over micro-batches with proper scaling, step once, return mean loss'''

    scale = 1 / len(micro_batches)
    optimizer.zero_grad()
    sum_loss = 0.0
    for x, y in micro_batches:
        loss = criterion(model(x), y)
        (loss * scale).backward()
        sum_loss += loss.item()
    optimizer.step()
    return sum_loss * scale
