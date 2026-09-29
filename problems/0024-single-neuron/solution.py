import torch
import torch.nn.functional as F


def single_neuron_model(
    features: list[list[float]], labels: list[int], weights: list[float], bias: float
) -> tuple[list[float], float]:
    """
    Simulates a single neuron with sigmoid activation for binary classification.

    Args:
        features: List of feature vectors (each a list of floats)
        labels: List of true binary labels
        weights: Neuron weights (one per feature)
        bias: Neuron bias term

    Returns:
        Tuple of (predicted probabilities rounded to 4 decimal places, MSE rounded to 4 decimal places)
    """
    # Your code here using PyTorch built-ins:
    # - torch.matmul() for linear combination
    # - torch.sigmoid() for activation
    # - torch.nn.functional.mse_loss() for MSE

    x = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor(labels)
    w = torch.tensor(weights, dtype=torch.float32)
    b = torch.tensor(bias, dtype=torch.float32)

    logits = torch.sigmoid(torch.matmul(x, w) + b)
    mse = F.mse_loss(logits, y)

    return logits.tolist(), round(mse.item(), 4)
