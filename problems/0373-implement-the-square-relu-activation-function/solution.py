import numpy as np


def square_relu(x: np.ndarray) -> dict:
    """
    Apply the Square ReLU activation function and compute its derivative.

    Args:
            x: Input numpy array of any shape

    Returns:
            Dictionary with 'output' and 'derivative' as numpy arrays
    """

    return {
        'output': np.round(np.maximum(x, 0) ** 2, 4),
        'derivative': np.round(np.maximum(x, 0) * 2, 4),
    }
