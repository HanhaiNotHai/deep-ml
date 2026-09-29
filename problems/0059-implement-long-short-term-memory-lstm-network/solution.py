import numpy as np
from numpy.typing import NDArray


def sigmoid(x: NDArray):
    return 1 / (1 + np.exp(-x))


class LSTM:
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Initialize weights and biases
        self.Wf = np.random.randn(hidden_size, input_size + hidden_size)
        self.Wi = np.random.randn(hidden_size, input_size + hidden_size)
        self.Wc = np.random.randn(hidden_size, input_size + hidden_size)
        self.Wo = np.random.randn(hidden_size, input_size + hidden_size)

        self.bf = np.zeros((hidden_size, 1))
        self.bi = np.zeros((hidden_size, 1))
        self.bc = np.zeros((hidden_size, 1))
        self.bo = np.zeros((hidden_size, 1))

    def forward(self, x, initial_hidden_state, initial_cell_state):
        """
        Processes a sequence of inputs and returns the hidden states, final hidden state, and final cell state.
        """
        h = initial_hidden_state
        c = initial_cell_state
        for xt in x:
            hx = np.vstack([h, xt[:, None]])
            f = sigmoid(self.Wf @ hx + self.bf)
            i = sigmoid(self.Wi @ hx + self.bi)
            c_tilde = np.tanh(self.Wc @ hx + self.bc)
            c = f * c + i * c_tilde
            o = sigmoid(self.Wo @ hx + self.bo)
            h = o * np.tanh(c)
        return o, h, c
