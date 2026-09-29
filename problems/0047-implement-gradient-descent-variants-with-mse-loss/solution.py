import numpy as np


def gradient_descent(X, y, weights, learning_rate, n_epochs, batch_size=1, method='batch'):
    """
    Perform gradient descent optimization.

    Args:
        X: Feature matrix of shape (m, n)
        y: Target values of shape (m,)
        weights: Initial weights of shape (n,)
        learning_rate: Step size for gradient descent
        n_epochs: Number of complete passes through the dataset
        batch_size: Size of batches for mini-batch gradient descent (default: 1)
        method: Type of gradient descent ('batch', 'stochastic', or 'mini_batch')

    Returns:
        Optimized weights
    """
    w = weights.copy()

    if method == 'batch':
        for _ in range(n_epochs):
            w -= learning_rate * 2 * ((X @ w - y)[:, None] * X).mean(axis=0)
        return w

    if method == 'stochastic':
        for _ in range(n_epochs):
            for xx, yy in zip(X, y):
                w -= learning_rate * 2 * (xx @ w - yy) * xx
        return w

    if method == 'mini_batch':
        for _ in range(n_epochs):
            for i in range(0, X.shape[0], batch_size):
                xx = X[i : i + batch_size]
                yy = y[i : i + batch_size]
                w -= learning_rate * 2 * ((xx @ w - yy)[:, None] * xx).mean(axis=0)
        return w
