import torch
from torch import Tensor


def softmax(z: Tensor):
    expz = torch.exp(z - z.max(dim=-1, keepdim=True).values)
    return expz / expz.sum(dim=-1, keepdim=True)


def focal_loss(logits: Tensor, targets: Tensor, gamma: float = 2.0):
    '''mean focal loss over the batch'''

    p = softmax(logits)
    p_t = p[range(len(targets)), targets]
    l = -((1 - p_t) ** gamma) * torch.log(p_t)
    return l.mean()
