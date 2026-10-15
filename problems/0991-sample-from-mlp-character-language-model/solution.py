from collections import deque

import numpy as np
from numpy.typing import NDArray


def softmax(x: NDArray):
    shifted = x - x.max(axis=-1, keepdims=True)
    expx = np.exp(shifted)
    return expx / expx.sum(axis=-1, keepdims=True)


def sample_mlp_lm(
    C: list[list[int]],
    W1: list[list[int]],
    b1: list[int],
    W2: list[list[int]],
    b2: list[int],
    block_size: int,
    itos: dict[int, str],
    seed: int,
    max_tokens: int,
):

    C, W1, b1, W2, b2 = map(lambda x: np.array(x, float), (C, W1, b1, W2, b2))
    rng = np.random.default_rng(seed)
    context = deque([0] * block_size)
    y = []
    for _ in range(max_tokens):
        emb: NDArray = C[context]
        emb_concat = emb.reshape(-1)
        h = np.tanh(emb_concat @ W1 + b1)
        logits = h @ W2 + b2

        probs = softmax(logits)
        i = rng.choice(len(probs), p=probs)

        if i == 0:
            break
        y.append(itos[i])

        context.popleft()
        context.append(i)

    return ''.join(y)
