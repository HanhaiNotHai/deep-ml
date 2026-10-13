import re
from typing import Literal


def tokenize(vocab: dict[str, int], text: str | list[int], mode: Literal['encode', 'decode']):
    """
    vocab: dict[str, int] containing '<|unk|>' and '<|endoftext|>'
    text: str (if mode='encode') or list[int] (if mode='decode')
    mode: 'encode' or 'decode'
    """

    if mode == 'encode':
        pattern = r"([,.:;?_!\"()']|--|\s)"
        tokens = [t.strip() for t in re.split(pattern, text) if t.strip()]
        unk_id = vocab['<|unk|>']
        return [vocab.get(t, unk_id) for t in tokens]

    inv_vocab = {i: t for t, i in vocab.items()}
    text = ' '.join(inv_vocab[i] for i in text)
    return re.sub(r"\s+([,.?!\"()'])", r"\1", text)
