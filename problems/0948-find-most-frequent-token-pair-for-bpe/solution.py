from collections import Counter
from itertools import chain, pairwise


def most_frequent_pair(sequences: list[list[int]]):
    """
    Args:
        sequences: list[list[int]] - list of token ID sequences
    Returns:
        tuple(int, int) or None - the most frequent adjacent pair, with ties
        broken by first appearance. Returns None if no pair exists.
    """

    cnt = Counter(chain(*map(pairwise, sequences)))
    return cnt.most_common(1)[0][0] if cnt else None
