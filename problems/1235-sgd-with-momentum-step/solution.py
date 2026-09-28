import torch


def momentum_step(w, grad, v, lr, mu):
    """One SGD-with-momentum step.

    Args:
        w: parameter tensor
        grad: gradient tensor (same shape as w)
        v: velocity tensor (same shape as w)
        lr: learning rate (float)
        mu: momentum coefficient (float)

    Returns:
        (w_new, v_new) tuple of tensors
    """

    v = mu * v + grad
    w -= lr * v
    return w, v
