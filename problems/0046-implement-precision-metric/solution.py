import numpy as np


def precision(y_true, y_pred):
    return ((y_true == 1) & (y_pred == 1)).sum() / (y_pred == 1).sum()
