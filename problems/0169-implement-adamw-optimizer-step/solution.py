import numpy as np
from numpy.typing import NDArray


def adamw_update(
    w: NDArray,
    g: NDArray,
    m: NDArray,
    v: NDArray,
    t: int,
    lr: float,
    beta1: float,
    beta2: float,
    epsilon: float,
    weight_decay: float,
):
    """
    Perform one AdamW optimizer step.
    Args:
      w: parameter vector (np.ndarray)
      g: gradient vector (np.ndarray)
      m: first moment vector (np.ndarray)
      v: second moment vector (np.ndarray)
      t: integer, current time step
      lr: float, learning rate
      beta1: float, beta1 parameter
      beta2: float, beta2 parameter
      epsilon: float, small constant
      weight_decay: float, weight decay coefficient
    Returns:
      w_new, m_new, v_new
    """

    m = beta1 * m + (1 - beta1) * g
    v = beta2 * v + (1 - beta2) * g * g

    m_hat = m / (1 - beta1**t)
    v_hat = v / (1 - beta2**t)

    w -= lr * weight_decay * w

    w -= lr * m_hat / (np.sqrt(v_hat) + epsilon)

    return w, m, v
