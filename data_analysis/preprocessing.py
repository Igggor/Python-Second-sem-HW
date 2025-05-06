from typing import Tuple
import numpy as np


def get_boxplot_outliers(
    data: np.ndarray,
    axis: int | None = None,  # Для многомерных данных (None = сжатие в 1D)
) -> np.ndarray:
    """Поиск выбросов с помощью метода boxplot (IQR)."""
    if axis is not None:
        data = data[:, axis]  # Выбираем конкретную ось (0 или 1)

    data_sorted = np.sort(data)
    n = len(data_sorted)

    q1 = data_sorted[int(n * 0.25)]
    q3 = data_sorted[int(n * 0.75)]
    epsilon = (q3 - q1) * 1.5

    lower_bound = q1 - epsilon
    upper_bound = q3 + epsilon

    outliers = np.where((data < lower_bound) | (data > upper_bound))[0]
    print(outliers)
    return outliers


def train_test_split(
    features: np.ndarray,
    targets: np.ndarray,
    train_ratio: float = 0.8,
    shuffle: bool = True,
    random_seed: int | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Разделяет данные на train и test с сохранением баланса классов.

    Параметры:
    ----------
    features : np.ndarray
        Массив признаков формы (n_samples, n_features).
    targets : np.ndarray
        Массив меток формы (n_samples,).
    train_ratio : float, optional
        Доля обучающей выборки (по умолчанию 0.8).
    shuffle : bool, optional
        Если True, данные перемешиваются (по умолчанию True).
    random_seed : int, optional
        Seed для генератора случайных чисел (по умолчанию None).

    Возвращает:
    -----------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        (train_features, train_labels, test_features, test_labels)
    """
    # Проверка входных данных
    if len(features) != len(targets):
        raise ValueError("features и targets должны иметь одинаковую длину")
    if not 0 <= train_ratio <= 1:
        raise ValueError("train_ratio должен быть в диапазоне [0, 1]")

    rng = np.random.default_rng(random_seed)
    unique_classes = np.unique(targets)
    train_features, test_features = [], []
    train_labels, test_labels = [], []

    for cls in unique_classes:
        cls_indices = np.where(targets == cls)[0]
        n_train = int(len(cls_indices) * train_ratio)

        if shuffle:
            rng.shuffle(cls_indices)  # Используем современный генератор

        train_features.append(features[cls_indices[:n_train]])
        train_labels.append(targets[cls_indices[:n_train]])
        test_features.append(features[cls_indices[n_train:]])
        test_labels.append(targets[cls_indices[n_train:]])

    return (
        np.vstack(train_features),
        np.concatenate(train_labels),
        np.vstack(test_features),
        np.concatenate(test_labels),
    )
