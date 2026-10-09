import torch
import torch.nn.functional as F

def overlapping_max_pool2d(x: torch.Tensor, kernel_size: int = 3, stride: int = 2) -> torch.Tensor:
    """
    Apply overlapping max pooling using PyTorch.
    Must match the ceil mode behavior described in the problem.
    """
    # Your code here
    pass