import torch
import torch.nn as nn


def train_neuron(
    features: torch.Tensor,
    labels: torch.Tensor,
    initial_weights: torch.Tensor,
    initial_bias: float,
    learning_rate: float,
    epochs: int,
) -> tuple[list[float], float, list[float]]:
    """
    Simulates a single neuron with sigmoid activation and trains it using
    backpropagation with MSE loss via SGD.

    Args:
        features: Input feature tensor of shape (n_samples, n_features)
        labels: Binary label tensor of shape (n_samples,)
        initial_weights: Initial weight tensor of shape (n_features,)
        initial_bias: Initial bias scalar
        learning_rate: Learning rate for SGD
        epochs: Number of training epochs

    Returns:
        Tuple of (updated_weights, updated_bias, mse_values) all rounded to 4 decimal places
    """

    w = initial_weights.clone().detach().requires_grad_()
    b = torch.tensor(initial_bias, dtype=w.dtype, requires_grad=True)

    mse_values = []
    for _ in range(epochs):
        loss = nn.functional.mse_loss(torch.sigmoid(features @ w + b), labels)
        loss.backward()
        with torch.inference_mode():
            w -= learning_rate * w.grad
            b -= learning_rate * b.grad
        w.grad.zero_()
        b.grad.zero_()

        mse_values.append(round(loss.item(), 4))

    updated_weights = list(map(lambda x: round(x, 4), w.tolist()))
    updated_bias = round(b.item(), 4)

    return updated_weights, updated_bias, mse_values
