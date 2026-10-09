from math import inf
from typing import Tuple

import torch


def early_stopping(val_losses: torch.Tensor, patience: int, min_delta: float) -> Tuple[int, int]:
    """
    Determine when to stop training early based on validation losses.

    Args:
        val_losses: A 1D tensor of validation losses for each epoch
        patience: Number of epochs without improvement before stopping
        min_delta: Minimum decrease in loss to qualify as an improvement

    Returns:
        Tuple of (stop_epoch, best_epoch)
    """

    best_loss = inf
    cnt = 0

    for epoch, loss in enumerate(val_losses):
        loss = loss.item()
        if best_loss - loss > min_delta:
            best_loss = loss
            best_epoch = epoch
            cnt = 0
        else:
            cnt += 1
            if cnt >= patience:
                break

    return epoch, best_epoch
