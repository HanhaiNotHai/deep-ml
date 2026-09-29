import numpy as np


def precision(y_true, y_pred):
    return ((y_true == 1) & (y_pred == 1)).sum() / (y_pred == 1).sum()


def recall(y_true, y_pred):
    return ((y_pred == 1) & (y_true == 1)).sum() / (y_true == 1).sum()


def f_score(y_true, y_pred, beta):
    """
    Calculate F-Score for a binary classification task.

    :param y_true: Numpy array of true labels
    :param y_pred: Numpy array of predicted labels
    :param beta: The weight of precision in the harmonic mean
    :return: F-Score rounded to three decimal places
    """
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return np.round((1 + beta**2) * p * r / ((beta**2 * p) + r), 3)
