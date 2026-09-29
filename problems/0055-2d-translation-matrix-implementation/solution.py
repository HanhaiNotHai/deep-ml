import numpy as np


def translate_object(points, tx, ty):
    T = np.array(
        [
            [1, 0, tx],
            [0, 1, ty],
            [0, 0, 1],
        ]
    )
    P = np.vstack([np.asarray(points).T, np.ones(len(points))])
    return (T @ P)[:2].T
