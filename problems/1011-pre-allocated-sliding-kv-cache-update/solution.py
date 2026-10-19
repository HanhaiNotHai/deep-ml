import numpy as np


def sliding_kv_cache_update(
    cache_k: list[list[list[list[int]]]],
    cache_v: list[list[list[list[int]]]],
    current_len: int,
    new_k: list[list[list[list[int]]]],
    new_v: list[list[list[list[int]]]],
    window_size: int,
):
    """
    Update a pre-allocated sliding-window KV cache and produce the causal mask
    for the new queries against the updated cache.

    Args:
        cache_k, cache_v: nested lists / arrays of shape
            (batch, num_heads, window_size, head_dim).
        current_len: int, number of valid entries currently in the cache (<= window_size).
        new_k, new_v: nested lists / arrays of shape
            (batch, num_heads, new_len, head_dim).
        window_size: int, maximum cache length.

    Returns:
        dict with keys 'cache_k', 'cache_v', 'current_len', 'mask'.
    """

    cache_k = np.array(cache_k, dtype=float, copy=True)
    cache_v = np.array(cache_v, dtype=float, copy=True)
    new_k = np.asarray(new_k)
    new_v = np.asarray(new_v)

    new_len = new_k.shape[2]
    updated_len = min(current_len + new_len, window_size)

    for cache, new in (cache_k, new_k), (cache_v, new_v):
        combined = np.concatenate((cache[:, :, :current_len, :], new), axis=2)
        recent = combined[:, :, -updated_len:, :] if updated_len else combined[:, :, :0, :]
        cache[:, :, :updated_len, :] = recent

    query_pos = np.maximum(updated_len - new_len + np.arange(new_len), 0)
    mask = (np.arange(updated_len)[None, :] <= query_pos[:, None]).astype(int)

    return {
        'cache_k': cache_k.tolist(),
        'cache_v': cache_v.tolist(),
        'current_len': updated_len,
        'mask': mask.tolist(),
    }
