import numpy as np


def xavier_init(fan_in: int, fan_out: int, mode: str = 'uniform', seed: int = 42):
    """
    Perform Xavier/Glorot weight initialization.

    Args:
        fan_in (int): Number of input units.
        fan_out (int): Number of output units.
        mode (str): 'uniform' or 'normal'.
        seed (int): Random seed for reproducibility.

    Returns:
        dict: Contains 'weights' (nested list), 'shape' (list), and 'param' (float).
    """

    np.random.seed(seed)

    shape = [fan_in, fan_out]

    if mode == 'uniform':
        param = np.sqrt(6 / (fan_in + fan_out))
        weights = np.random.uniform(-param, param, shape)
    elif mode == 'normal':
        param = np.sqrt(2 / (fan_in + fan_out))
        weights = np.random.randn(*shape) * param
    else:
        raise ValueError("mode must be 'uniform' or 'normal'")

    return {
        'weights': weights.tolist(),
        'shape': shape,
        'param': param,
    }
