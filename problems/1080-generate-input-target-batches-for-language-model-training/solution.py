import numpy as np


def get_batch(data: np.ndarray, block_size: int, batch_size: int, seed: int) -> np.ndarray:
    starts = np.random.default_rng(seed).integers(0, data.shape[0] - block_size, batch_size)
    indices = starts[:, None] + np.arange(block_size)
    x = data[indices]
    y = data[indices + 1]
    return np.stack((x, y))
