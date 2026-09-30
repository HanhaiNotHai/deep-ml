from typing import Callable, Tuple

import torch


def tensorenumerate(x: torch.Tensor):
    shape = x.shape
    for flat_idx, val in enumerate(x.flatten()):
        idx = torch.unravel_index(torch.tensor(flat_idx), shape)
        yield tuple(i.item() for i in idx), val


def numerical_gradient_check(
    f: Callable, x: torch.Tensor, analytical_grad: torch.Tensor, epsilon: float = 1e-7
) -> Tuple[torch.Tensor, float]:
    """
    Perform numerical gradient checking using centered finite differences.

    Args:
        f: A function that takes a torch.Tensor and returns a scalar
        x: torch.Tensor, the point at which to check gradient
        analytical_grad: torch.Tensor, the analytically computed gradient
        epsilon: float, small value for finite difference approximation

    Returns:
        tuple: (numerical_grad, relative_error)
    """

    coeff = 1 / (2 * epsilon)

    numerical_grad = torch.empty_like(x)
    plus = x.clone()
    minus = x.clone()

    for idx, val in tensorenumerate(x):
        plus[idx] += epsilon
        minus[idx] -= epsilon

        numerical_grad[idx] = (f(plus) - f(minus)) * coeff

        plus[idx] = minus[idx] = val

    relative_error = torch.norm(numerical_grad - analytical_grad) / (
        torch.norm(numerical_grad) + torch.norm(analytical_grad)
    )
    return numerical_grad, relative_error.item()
