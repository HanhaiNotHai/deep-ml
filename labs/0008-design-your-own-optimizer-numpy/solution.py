import numpy as np


def optimizer_step(param, grad, state, lr):
    '''
    Update a parameter using its gradient.

    Args:
        param: numpy array - current parameter values (any shape)
        grad: numpy array - gradient of loss w.r.t. param (same shape)
        state: dict - persists between calls, use to store any needed values
                      Example: state = {'step': 5, 'momentum': np.array([...])}
                      First call: state = {} (empty dict)
        lr: float - learning rate

    Returns:
        new_param: numpy array - updated parameter (must be same shape as param)
        state: dict - updated state dictionary
    '''

    if not state:
        state = {
            't': 1,
            'm': np.zeros_like(param),
            'v': np.zeros_like(param),
            'beta1': 0.9,
            'beta2': 0.999,
            'eps': 1e-8,
            'weight_decay': 0.01,
        }

    t = state['t']
    m = state['m']
    v = state['v']
    beta1 = state['beta1']
    beta2 = state['beta2']
    eps = state['eps']
    weight_decay = state['weight_decay']

    new_param = param - lr * weight_decay * param

    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * grad * grad

    m_hat = m / (1 - beta1**t)
    v_hat = v / (1 - beta2**t)

    new_param -= lr * m_hat / (np.sqrt(v_hat) + eps)

    t += 1

    state['t'] = t
    state['m'] = m
    state['v'] = v

    return new_param, state
