from functools import partial

import torch


def grad_wss(w_list, x_list):
    """Build w (requires_grad) and x from lists, compute
    loss = 0.5 * sum((w * x)**2), backward, return w.grad
    as a list of floats rounded to 4 decimals.
    """

    w = torch.tensor(w_list, requires_grad=True)
    x = torch.tensor(x_list)

    L = ((w * x) ** 2).sum() / 2

    L.backward()

    return list(map(partial(round, ndigits=4), w.grad.tolist()))
