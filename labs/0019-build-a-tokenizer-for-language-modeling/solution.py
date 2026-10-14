from collections import Counter
from itertools import pairwise


def train_tokenizer(corpus: list[str], vocab_size: int):
    """
    Build a tokenizer from the training corpus for downstream language modeling.

    The harness will use your tokenizer to train a bigram language model.
    Better tokenizers create more predictable token sequences, leading to
    lower bits-per-character on held-out data.

    Args:
        corpus: list[str] -- training documents (names like 'emma', 'olivia', ...)
        vocab_size: int   -- maximum vocabulary size (256)

    Returns:
        encode: callable, str -> list[int]
            Convert text to token IDs. All IDs must be in [0, vocab_size).
        decode: callable, list[int] -> str
            Convert token IDs back to text. Must satisfy decode(encode(s)) == s.
    """

    alphabet = sorted({char for text in corpus for char in text})
    pieces = alphabet[:]
    char2id = {char: i for i, char in enumerate(alphabet)}
    words = Counter(tuple(char2id[char] for char in text) for text in corpus)
    merges = []

    def merge(tokens: tuple[int, ...], pair: tuple[int, int], new_id: int):
        result: list[int] = []
        i = 0

        while i < len(tokens):
            if i + 1 < len(tokens) and (tokens[i], tokens[i + 1]) == pair:
                result.append(new_id)
                i += 2
            else:
                result.append(tokens[i])
                i += 1

        return result

    while len(pieces) < vocab_size:
        cnt: Counter[tuple[int, int]] = Counter()
        for tokens, freq in words.items():
            for pair in pairwise(tokens):
                cnt[pair] += freq
        if not cnt:
            break

        pair, freq = cnt.most_common(1)[0]
        if freq < 2:
            break

        new_id = len(pieces)
        pieces.append(pieces[pair[0]] + pieces[pair[1]])
        merges.append((pair, new_id))

        new_words: Counter[tuple[int, ...]] = Counter()
        for tokens, freq in words.items():
            new_words[tuple(merge(tokens, pair, new_id))] += freq
        words = new_words

    def encode(text: str) -> list[int]:
        tokens = [char2id[char] for char in text]
        for pair, new_id in merges:
            tokens = merge(tokens, pair, new_id)
        return tokens

    def decode(ids: list[int]) -> str:
        return ''.join(pieces[i] for i in ids)

    return encode, decode
