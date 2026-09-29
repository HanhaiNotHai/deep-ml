import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset


def train_model(
    model: nn.Module,
    X_train: Tensor,
    y_train: Tensor,
    X_val: Tensor,
    y_val: Tensor,
    epochs: int,
    batch_size: int,
    lr: float,
):
    """
    Train a PyTorch model and return training history.

    This is the standard PyTorch training pattern you'll use everywhere.
    Now you can use torch.optim to handle the gradient updates!

    Args:
        model: nn.Module to train
        X_train: training features, shape (N, ...)
        y_train: training labels, shape (N,)
        X_val: validation features, shape (M, ...)
        y_val: validation labels, shape (M,)
        epochs: number of training epochs
        batch_size: mini-batch size
        lr: learning rate

    Returns:
        history: List of dicts, one per epoch, with keys:
            - 'epoch': epoch number (starting from 1)
            - 'train_loss': average training loss for the epoch
            - 'val_loss': validation loss after the epoch
            - 'val_accuracy': validation accuracy after the epoch

    Steps:
        1. Create optimizer: optim.Adam(model.parameters(), lr=lr)
        2. Create loss function: nn.CrossEntropyLoss()
        3. For each epoch:
            a. Shuffle training data
            b. Loop over mini-batches:
                - optimizer.zero_grad()
                - Forward pass
                - Compute loss
                - loss.backward()
                - optimizer.step()
            c. Compute validation accuracy
            d. Append metrics to history
        4. Return history

    Hints:
        - torch.randperm(n) gives a random permutation for shuffling
        - Use model.train() before training, model.eval() before validation
        - Use torch.no_grad() during validation
        - logits.argmax(dim=1) gives predicted classes
    """

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size, shuffle=True)

    optimizer = optim.AdamW(model.parameters(), lr)
    criterion = nn.CrossEntropyLoss()

    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for x, y in train_loader:
            optimizer.zero_grad()
            loss: Tensor = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        avg_loss = sum(losses) / len(losses)

        model.eval()
        with torch.inference_mode():
            logits: Tensor = model(X_val)
            predictions = logits.argmax(dim=1)
            correct = (predictions == y_val).sum().item()
            acc = correct / y_val.shape[0]

        history.append(
            {
                'epoch': epoch,
                'train_loss': avg_loss,
                'val_accuracy': acc,
            }
        )

    return history
