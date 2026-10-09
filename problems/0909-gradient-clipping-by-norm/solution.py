import torch


def clip_grad_norm(parameters: list[torch.Tensor], max_norm: float) -> float:
    '''compute total grad norm, scale in-place if it exceeds max_norm, return original norm'''

    grads = [p.grad for p in parameters if p.grad is not None]

    if not grads:
        return 0.0

    total_norm = torch.norm(torch.stack([g.detach().norm() for g in grads])).item()

    if total_norm > max_norm:
        scale = max_norm / total_norm
        with torch.no_grad():
            for g in grads:
                g.mul_(scale)

    return total_norm
