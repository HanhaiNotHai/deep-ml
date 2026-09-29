import torch


def activation_derivatives(x: float) -> dict[str, float]:
    """
    Compute the derivatives of Sigmoid, Tanh, and ReLU at a given point x
    using PyTorch autograd.

    Args:
        x: Input value

    Returns:
        Dictionary with keys 'sigmoid', 'tanh', 'relu' and their derivative values
    """

    X = torch.tensor(x, dtype=torch.float32, requires_grad=True)

    torch.sigmoid(X).backward()
    s = X.grad.item()

    X.grad.zero_()

    torch.tanh(X).backward()
    t = X.grad.item()

    X.grad.zero_()

    torch.relu(X).backward()
    r = X.grad.item()

    return {
        'sigmoid': s,
        'tanh': t,
        'relu': r,
    }
