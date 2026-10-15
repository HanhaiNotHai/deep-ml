import numpy as np


def add_positional_embeddings(
    token_embeddings: np.ndarray, pos_embedding_matrix: np.ndarray
) -> np.ndarray:
    """Add absolute positional embeddings to token embeddings."""

    return token_embeddings + pos_embedding_matrix[None, : token_embeddings.shape[1]]
