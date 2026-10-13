def vocab_size(tokens: list[str], special_tokens: list[str] = None):
    """Return the vocabulary size from a list of preprocessed tokens.

    Args:
        tokens: list[str] of preprocessed tokens
        special_tokens: optional list[str] of special tokens to include
    Returns:
        int: number of unique tokens in the combined vocabulary
    """

    vocab = set(tokens)
    if special_tokens is not None:
        vocab |= set(special_tokens)
    return len(vocab)
