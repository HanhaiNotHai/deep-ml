from typing import Callable

import numpy as np
from numpy.typing import NDArray


def generate_greedy(
    model: Callable[[NDArray], NDArray], idx: list[int], max_new_tokens: int, context_size: int
):
    for _ in range(max_new_tokens):
        idx.append(model(np.array([idx[-context_size:]]))[0, -1].argmax().item())
    return idx
