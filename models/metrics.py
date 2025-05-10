import numpy as np


def accuracy(true_targets: np.ndarray, prediction: np.ndarray) -> float:
    """Вычисляет точность классификации.

    Args:
        true_targets: Истинные метки (n_samples,)
        prediction: Предсказанные метки (n_samples,)

    Returns:
        Точность предсказаний (0-1)
    """
    return np.mean(true_targets == prediction)
