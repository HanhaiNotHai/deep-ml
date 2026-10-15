import numpy as np


def embedding_via_one_hot(token_ids: list[int], W: np.ndarray):
    """
    Compute token embeddings via one-hot encoding and matrix multiplication.

    Args:
        token_ids: list or 1D array of integer token IDs
        W: numpy array of shape (vocab_size, embed_dim)

    Returns:
        numpy array of shape (len(token_ids), embed_dim)
    """

    return np.eye(W.shape[0])[token_ids] @ W
