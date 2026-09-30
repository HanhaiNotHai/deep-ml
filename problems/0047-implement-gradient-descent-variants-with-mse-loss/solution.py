import torch
from torch import Tensor


def gradient(X: Tensor, y: Tensor, w: Tensor):
    torch.nn.functional.mse_loss(X @ w, y).backward()
    return w.grad


def gd(w: Tensor, lr: float, g: Tensor):
    with torch.no_grad():
        return (w - lr * g).requires_grad_()


def bgd(X: Tensor, y: Tensor, w: Tensor, lr: float, *args):
    return gd(w, lr, gradient(X, y, w))


def sgd(X: Tensor, y: Tensor, w: Tensor, lr: float, *args):
    for xx, yy in zip(X, y):
        w = gd(w, lr, gradient(xx[None], yy[None], w))
    return w


def mbgd(X: Tensor, y: Tensor, w: Tensor, lr: float, bs: int, *args):
    for i in range(0, X.shape[0], bs):
        xx = X[i : i + bs]
        yy = y[i : i + bs]
        w = gd(w, lr, gradient(xx[None], yy[None], w))
    return w


METHOD2GD = {'batch': bgd, 'stochastic': sgd, 'mini_batch': mbgd}


def gradient_descent(
    X: Tensor,
    y: Tensor,
    weights: Tensor,
    learning_rate: float,
    n_epochs: int,
    batch_size: int = 1,
    method: str = 'batch',
) -> Tensor:
    """
    Implements three variants of gradient descent: Batch, Stochastic, and Mini-Batch.
    Uses Mean Squared Error (MSE) as the loss function.

    Args:
        X: Feature matrix of shape (m, n)
        y: Target values of shape (m,)
        weights: Initial weights of shape (n,)
        learning_rate: Step size for gradient descent
        n_epochs: Number of complete passes through the dataset
        batch_size: Size of batches for mini-batch gradient descent (default: 1)
        method: Type of gradient descent ('batch', 'stochastic', or 'mini_batch')

    Returns:
        Optimized weights as a tensor
    """

    gd_fn = METHOD2GD[method]
    for _ in range(n_epochs):
        weights = gd_fn(X, y, weights.requires_grad_(), learning_rate, batch_size)
    return weights
