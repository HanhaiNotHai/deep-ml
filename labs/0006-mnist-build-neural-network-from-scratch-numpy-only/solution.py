# --- cell 1 ---
import numpy as np
from numpy.typing import NDArray


def softmax(z: NDArray):
    expz = np.exp(z - z.max(axis=-1, keepdims=True))
    return expz / expz.sum(axis=-1, keepdims=True)


class NeuralNetwork:
    '''
    Build a neural network from scratch using only NumPy.
    Required architecture: 784 → 128 (ReLU) → 10 (Softmax)
    '''

    def __init__(
        self,
        input_size: int = 784,
        hidden_size: int = 128,
        output_size: int = 10,
        lr: float = 0.01,
    ):
        '''
        Initialize network parameters.
        Use small random initialization (e.g., Xavier/He initialization).
        '''

        self.lr = lr
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2 / input_size)
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2 / hidden_size)
        self.b2 = np.zeros(output_size)

    def forward(self, X: NDArray):
        '''
        Forward pass through the network.

        Args:
            X: Input batch, shape (N, 784)

        Returns:
            probs: Class probabilities, shape (N, 10)

        Must cache intermediate values for backward pass!
        '''

        self.z1 = X @ self.W1 + self.b1
        self.h1 = np.maximum(self.z1, 0)
        self.z2 = self.h1 @ self.W2 + self.b2
        return softmax(self.z2)

    def backward(self, X: NDArray, y: NDArray, probs: NDArray):
        '''
        Backward pass - compute gradients for all parameters.

        Args:
            X: Input batch, shape (N, 784)
            y: True labels, shape (N,)
            probs: Predicted probabilities from forward pass, shape (N, 10)

        Returns:
            loss: Scalar cross-entropy loss

        Must update self.W1, self.b1, self.W2, self.b2 using computed gradients!
        '''

        N = len(y)

        loss = -np.log(probs[range(N), y] + 1e-12).mean()

        dz2 = probs.copy()
        dz2[range(len(y)), y] -= 1
        dz2 /= N

        dW2 = self.h1.T @ dz2
        db2 = dz2.sum(axis=0)
        dh1 = dz2 @ self.W2.T

        dz1 = np.where(self.z1 > 0, dh1, 0)

        dW1 = X.T @ dz1
        db1 = dz1.sum(axis=0)

        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

        return loss

    def train_step(self, X: NDArray, y: NDArray):
        '''
        Complete training step: forward + backward + update.

        Args:
            X: Input batch, shape (N, 784)
            y: True labels, shape (N,)

        Returns:
            loss: Scalar loss value
        '''

        probs = self.forward(X)
        loss = self.backward(X, y, probs)
        return loss

    def predict(self, X: NDArray):
        '''
        Predict class labels.

        Args:
            X: Input batch, shape (N, 784)

        Returns:
            predictions: Predicted class labels, shape (N,)
        '''

        probs = self.forward(X)
        return np.argmax(probs, axis=1)