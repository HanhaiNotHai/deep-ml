def pad_and_truncate(sequences: list[list[int]], pad_token_id: int, max_length: int | None = None):
    """
    Pad and/or truncate tokenized sequences to a uniform length.

    Args:
        sequences: list of lists of int token ids
        pad_token_id: int, token id used for padding
        max_length: int or None, target length. If None, use the longest sequence length.

    Returns:
        list of lists of int token ids, each of length max_length.
    """

    if not sequences:
        return []

    if max_length is None:
        max_length = max(map(len, sequences))

    sequences = [seq[:max_length] for seq in sequences]
    sequences = [seq + [pad_token_id] * (max_length - len(seq)) for seq in sequences]
    return sequences
