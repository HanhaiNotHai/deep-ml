import math
from functools import partial


def sigmoid(z: float):
    return 1 / (1 + math.exp(-z))


def backprop_chain(a0: float, y: float, w: list[float], b: list[float]):
    """
    Compute gradients of C = (a_L - y)^2 with respect to each weight and bias
    in a chain network where every layer has a single sigmoid neuron.

    Args:
        a0: input activation (scalar)
        y: target value (scalar)
        w: list of L weights
        b: list of L biases

    Returns:
        Dictionary with keys 'dW' and 'dB', each a list of L floats rounded to 6 places.
    """

    L = len(w)

    w.insert(0, 0.0)
    b.insert(0, 0.0)
    a = [a0] * (L + 1)
    z = [0.0] * (L + 1)
    for l in range(1, L + 1, 1):
        z[l] = w[l] * a[l - 1] + b[l]
        a[l] = sigmoid(z[l])

    da = [2 * (a[-1] - y)] * (L + 1)
    dz = [0.0] * (L + 1)
    dW = [0.0] * (L + 1)
    dB = [0.0] * (L + 1)
    for l in range(L, 0, -1):
        dz[l] = da[l] * a[l] * (1 - a[l])
        dW[l] = dz[l] * a[l - 1]
        dB[l] = dz[l]
        da[l - 1] = dz[l] * w[l]

    return {
        'dW': list(map(partial(round, ndigits=6), dW[1:])),
        'dB': list(map(partial(round, ndigits=6), dB[1:])),
    }
