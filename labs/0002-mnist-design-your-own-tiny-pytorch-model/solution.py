import torch
import torch.nn as nn
from torch import Tensor


def build_model() -> nn.Module:
    """
    Return a tiny nn.Module for MNIST classification (10 classes).
    IMPORTANT: If total trainable params > 2048, final accuracy will be set to 0.
    Tip: Consider very small convs, global average pooling, and tiny linear head.
    """

    class TinyNet(nn.Module):

        def __init__(self):
            super().__init__()

            self.model = nn.Sequential(
                # Extract local patterns
                nn.Conv2d(1, 16, kernel_size=3, padding=1, bias=True),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                # Depthwise convolution
                nn.Conv2d(16, 16, kernel_size=3, padding=1, groups=16, bias=True),
                nn.ReLU(inplace=True),
                # Pointwise expasion
                nn.Conv2d(16, 32, kernel_size=1, bias=True),
                nn.ReLU(inplace=True),
                #
                nn.AdaptiveAvgPool2d((2, 2)),
                #
                nn.Flatten(1),
                #
                nn.Linear(128, 10),
            )

        def forward(self, x: Tensor):
            return self.model(x)

    return TinyNet()
