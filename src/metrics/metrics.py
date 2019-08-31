import numpy as np


def accurracy(pred_y, labels):
    assert len(pred_y) == len(labels)
    return np.sum(pred_y == labels) * 1.0 / len(pred_y)
