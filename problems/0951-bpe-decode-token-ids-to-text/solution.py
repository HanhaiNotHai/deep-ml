import re


def bpe_decode(ids: list[int], vocab: dict[int, str]):
    """
    Args:
        ids: list[int] - token IDs to decode
        vocab: dict[int, str] - mapping from token ID to token string
    Returns:
        str - the decoded text
    """

    return ''.join(map(lambda i: re.sub(f'^G', ' ', vocab[i]), ids))
