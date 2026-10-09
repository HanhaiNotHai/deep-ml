import torch
from torch import Tensor


def cross_entropy(logits: Tensor, targets: Tensor):
    '''numerically stable mean cross-entropy'''

    log_probs = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    return -log_probs.gather(dim=-1, index=targets[..., None]).mean()
