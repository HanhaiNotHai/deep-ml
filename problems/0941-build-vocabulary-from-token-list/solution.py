def build_vocab(tokens: list[str]):
    """
    Build a vocabulary dictionary from a list of tokens.

    Args:
        tokens: list of string tokens

    Returns:
        Dict mapping each unique token (sorted) to a unique integer ID starting from 0.
    """

    return {t: i for i, t in enumerate(sorted(set(tokens)))}
