import numpy as np


def token_embedding_lookup(
    vocab_size: int, embed_dim: int, token_ids: list, seed: int = 0
) -> list:
    """
    Build a random embedding table of shape (vocab_size, embed_dim) using
    np.random.default_rng(seed).standard_normal(...), then return the rows
    corresponding to token_ids as a nested list.
    """

    w = np.random.default_rng(seed).standard_normal((vocab_size, embed_dim))
    return [w[i].tolist() for i in token_ids]
