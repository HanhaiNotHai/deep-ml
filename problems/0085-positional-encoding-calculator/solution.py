import numpy as np


def pos_encoding(position: int, d_model: int):
    pos = np.arange(position)[:, None]
    i = np.arange(d_model)[None]

    angles = pos / np.power(10000, 2 * (i // 2) / d_model)

    pos_encoding = np.empty((position, d_model))
    pos_encoding[:, 0::2] = np.sin(angles[:, 0::2])
    pos_encoding[:, 1::2] = np.cos(angles[:, 1::2])
    return pos_encoding.astype(np.float16)
