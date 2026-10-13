import re


def encode(text: str, vocab: dict[str, int]):
    pattern = r"([,.:;?_!\"()']|--|\s)"
    tokens = [t.strip() for t in re.split(pattern, text) if t.strip()]
    return [vocab[t] for t in tokens]


def decode(ids: list[int], vocab: dict[str, int]):
    inv_vocab = {i: t for t, i in vocab.items()}
    text = ' '.join(inv_vocab[i] for i in ids)
    return re.sub(r"\s+([,.?!\"()'])", r"\1", text)
