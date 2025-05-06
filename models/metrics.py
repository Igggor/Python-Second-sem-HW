import numpy as np


def accuracy(true_targets: np.ndarray, prediction: np.ndarray) -> float:
    """Точность классификации"""
    return np.mean(true_targets == prediction)
