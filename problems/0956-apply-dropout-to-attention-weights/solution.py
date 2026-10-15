import numpy as np


def attention_dropout(
    attn_weights: list[list[float]],
    values: list[list[float]],
    dropout_rate: float,
    mask: list[list[int]],
):
    """Apply dropout mask to attention weights and compute context vectors."""

    return np.where(mask, attn_weights, 0) / (1 - dropout_rate) @ values
