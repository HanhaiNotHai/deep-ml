import numpy as np


def reduce_broadcast_grad(upstream_grad: np.ndarray, operand_shape: tuple[int]):
    """
    Reduce an upstream gradient back to the shape of a broadcasted operand.

    Args:
        upstream_grad: array of shape (N, D)
        operand_shape: either (1, D) or (N, 1)

    Returns:
        numpy array with shape exactly equal to operand_shape
    """

    return upstream_grad.sum(axis=operand_shape.index(1), keepdims=True)
