import math

import torch
from torch import Tensor


def kv_cache_attention_step(
    x_new: Tensor,
    W_Q: Tensor,
    W_K: Tensor,
    W_V: Tensor,
    cache: tuple[Tensor, Tensor] | None,
) -> tuple:
    """
    Perform a single attention step with KV caching.

    Args:
        x_new: New token embedding, shape (d_model,)
        W_Q: Query projection matrix, shape (d_model, d_k)
        W_K: Key projection matrix, shape (d_model, d_k)
        W_V: Value projection matrix, shape (d_model, d_v)
        cache: Tuple (K_cache, V_cache) of tensors or None if first step

    Returns:
        Tuple (output, updated_cache) where output is shape (d_v,)
        and updated_cache is (K_new, V_new)
    """

    x = x_new[None]
    q, k, v = map(lambda W: x @ W, (W_Q, W_K, W_V))

    if cache is None:
        K, V = k, v
    else:
        K_cache, V_cache = cache
        K = torch.concat((K_cache, k))
        V = torch.concat((V_cache, v))

    o = torch.softmax(q @ K.T / math.sqrt(K.shape[-1]), dim=-1) @ V

    return o[0], (K, V)
