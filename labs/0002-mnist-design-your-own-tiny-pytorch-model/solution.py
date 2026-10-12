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
                nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(8),
                nn.ReLU(inplace=True),
                # Depthwise convolution
                nn.Conv2d(
                    in_channels=8, out_channels=8, kernel_size=3, padding=1, groups=8, bias=False
                ),
                nn.BatchNorm2d(8),
                nn.ReLU(inplace=True),
                # Pointwise expasion
                nn.Conv2d(in_channels=8, out_channels=16, kernel_size=1, bias=False),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                #
                nn.AdaptiveAvgPool2d(1),
                #
                nn.Flatten(1),
                #
                nn.Linear(16, 10),
            )

        def forward(self, x: Tensor):
            return self.model(x)

    return TinyNet()
