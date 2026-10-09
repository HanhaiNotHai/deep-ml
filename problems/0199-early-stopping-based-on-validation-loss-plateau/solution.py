from math import inf

import torch


def early_stopping(
    val_losses: torch.Tensor, patience: int = 5, min_delta: float = 0.0
) -> torch.Tensor:
    """
    Determine at each epoch whether training should stop based on validation loss.

    Args:
        val_losses: Tensor of validation losses at each epoch
        patience: Number of epochs to wait for improvement before stopping
        min_delta: Minimum change in validation loss to qualify as improvement

    Returns:
        Tensor of booleans indicating whether to stop at each epoch
    """

    eps = torch.finfo(val_losses.dtype).eps

    best_loss = inf
    cnt = 0
    result = []

    for loss in val_losses.tolist():
        if best_loss - loss > min_delta + eps:
            best_loss = loss
            cnt = 0
        else:
            cnt += 1

        result.append(cnt >= patience)

    return torch.tensor(result)
