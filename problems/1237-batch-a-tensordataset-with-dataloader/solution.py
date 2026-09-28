import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset


def batch_stats(X: Tensor, y: Tensor):
    """Wrap X and y in TensorDataset + DataLoader(batch_size=4, shuffle=False).

    Return (num_batches, first_batch_X_shape_tuple).
    """

    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)

    num_batches = 0
    first_batch_X_shape_tuple = None
    for batch_X, batch_Y in dataloader:
        num_batches += 1
        if first_batch_X_shape_tuple is None:
            first_batch_X_shape_tuple = tuple(batch_X.shape)

    return num_batches, first_batch_X_shape_tuple
