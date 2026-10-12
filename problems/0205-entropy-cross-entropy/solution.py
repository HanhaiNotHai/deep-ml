import torch


def entropy_and_cross_entropy(P: torch.Tensor, Q: torch.Tensor) -> tuple[float, float]:
    """
    Compute entropy of P and cross-entropy between P and Q.

    Args:
        P: True probability distribution (torch.Tensor)
        Q: Predicted probability distribution (torch.Tensor)

    Returns:
        Tuple of (entropy H(P), cross-entropy H(P,Q))
    """

    return tuple(
        map(
            lambda p, q: torch.where(p > 0, -p * torch.log(q), torch.tensor(0)).sum().item(),
            (P, P),
            (P, Q),
        )
    )
