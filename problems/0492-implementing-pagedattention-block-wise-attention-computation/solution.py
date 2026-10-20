import numpy as np
from numpy.typing import NDArray


def softmax(x: NDArray):
    expx = np.exp(x - x.max(axis=-1, keepdims=True))
    return expx / expx.sum(axis=-1, keepdims=True)


def paged_attention(
    query: NDArray,
    key_cache: NDArray,
    value_cache: NDArray,
    block_table: list[int],
    context_len: int,
):
    """
    Perform scaled dot-product attention with paged KV cache.

    Args:
        query:       (num_heads, head_dim) query for a single token
        key_cache:   (num_physical_blocks, block_size, num_heads, head_dim)
        value_cache: (num_physical_blocks, block_size, num_heads, head_dim)
        block_table: list of physical block indices (logical -> physical)
        context_len: number of valid KV tokens in the sequence

    Returns:
        (num_heads, head_dim) attention output, rounded to 4 decimal places
    """

    q = query[None]
    k, v = map(
        lambda x: x[block_table].reshape(-1, *x.shape[-2:])[:context_len], (key_cache, value_cache)
    )

    q, k, v = map(lambda x: x.swapaxes(-2, -3), (q, k, v))

    s = np.einsum('...ld,...id->...li', q, k) / np.sqrt(q.shape[-1])
    a = softmax(s)
    o: NDArray = np.einsum('...li,...id->...ld', a, v)

    o = o.swapaxes(-2, -3)

    return o[0]
