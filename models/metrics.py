import numpy as np


def accuracy(true_targets: np.ndarray, prediction: np.ndarray) -> float:
    return np.mean(true_targets == prediction)
