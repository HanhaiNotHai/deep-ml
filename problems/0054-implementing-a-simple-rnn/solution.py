import numpy as np


def rnn_forward(
    input_sequence: list[list[float]],
    initial_hidden_state: list[float],
    Wx: list[list[float]],
    Wh: list[list[float]],
    b: list[float],
) -> list[float]:
    X = np.asarray(input_sequence)
    h = np.asarray(initial_hidden_state)
    Wx = np.asarray(Wx)
    Wh = np.asarray(Wh)
    b = np.asarray(b)
    for x in X:
        h = np.tanh(Wx @ x + Wh @ h + b)
    return h
