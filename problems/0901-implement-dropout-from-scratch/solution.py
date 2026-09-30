import torch


def dropout(x: torch.Tensor, p: float, training: bool) -> torch.Tensor:
    '''implement inverted dropout'''

    if not training:
        return x
    return x * (torch.rand_like(x) > p) / (1 - p)
