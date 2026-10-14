from collections import Counter
from itertools import pairwise


def byte_pair_encoding(corpus: dict[str, int], num_merges: int):
    """
    Train a BPE tokenizer on the given corpus.

    Args:
        corpus: Dictionary mapping space-separated token sequences to their frequencies.
                Example: {"l o w </w>": 5, "n e w </w>": 6}
        num_merges: Number of merge operations to perform.

    Returns:
        List of tuples, where each tuple contains the two tokens that were merged.
        Example: [('l', 'o'), ('lo', 'w')]
    """

    words = [(seq.split(), freq) for seq, freq in corpus.items()]
    merges = []

    for _ in range(num_merges):
        cnt: Counter[tuple[str, str], int] = Counter()
        for tokens, freq in words:
            for pair in pairwise(tokens):
                cnt[pair] += freq

        if not cnt:
            break

        best_pair = cnt.most_common(1)[0][0]
        merges.append(best_pair)
        merged_token = ''.join(best_pair)

        new_words = []
        for tokens, freq in words:
            merged = []
            i = 0
            while i < len(tokens):
                if i + 1 < len(tokens) and (tokens[i], tokens[i + 1]) == best_pair:
                    merged.append(merged_token)
                    i += 2
                else:
                    merged.append(tokens[i])
                    i += 1
            new_words.append((merged, freq))
        words = new_words

    return merges
