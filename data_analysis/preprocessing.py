from typing import Tuple, Callable, Any
import numpy as np


def get_boxplot_outliers(
    data: np.ndarray,
    key: Callable[[Any], Any] = None,
) -> np.ndarray:
    """
    Поиск выбросов в n-мерном массиве с помощью метода boxplot (IQR).

    Параметры:
    ----------
    data : np.ndarray
        Входные данные (может быть n-мерным)
    key : Callable, optional
        Функция для сортировки (если None - обычная сортировка)
    axis : int или None, optional
        Ось для анализа (None - обрабатывает все оси отдельно и объединяет результаты)

    Возвращает:
    -----------
    np.ndarray
        Уникальные индексы выбросов
    """
    if data.ndim == 1:
        # 1D-случай реализовал как было раньше
        if key is not None:
            sorted_indices = np.argsort([key(x) for x in data])
            data_sorted = data[sorted_indices]
        else:
            data_sorted = np.sort(data)

        n = len(data_sorted)
        q1 = data_sorted[int(n * 0.25)]
        q3 = data_sorted[int(n * 0.75)]
        epsilon = (q3 - q1) * 1.5

        lower_bound = q1 - epsilon
        upper_bound = q3 + epsilon

        return np.where((data < lower_bound) | (data > upper_bound))[0]

    else:
        # Многомерный случай
        all_outliers = []

        axes_to_check = range(data.shape[1])

        for ax in axes_to_check:
            axis_data = data[:, ax]

            # Применяю алгоритм для 1D-случая
            if key is not None:
                sorted_indices = np.argsort([key(x) for x in axis_data])
                data_sorted = axis_data[sorted_indices]
            else:
                data_sorted = np.sort(axis_data)

            n = len(data_sorted)
            q1 = data_sorted[int(n * 0.25)]
            q3 = data_sorted[int(n * 0.75)]
            epsilon = (q3 - q1) * 1.5

            lower_bound = q1 - epsilon
            upper_bound = q3 + epsilon

            outliers = np.where((axis_data < lower_bound) | (axis_data > upper_bound))[0]
            all_outliers.extend(outliers.tolist())

        # Возвращаем только уникальные индексы
        return np.unique(all_outliers)


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
            # Использую генератор, который я создал выше
            rng.shuffle(cls_indices)

        train_features.append(features[cls_indices[:n_train]])
        train_labels.append(targets[cls_indices[:n_train]])
        test_features.append(features[cls_indices[n_train:]])
        test_labels.append(targets[cls_indices[n_train:]])

    return (
        np.vstack(train_features),  # Объединение массивов вертикально (по первой оси)
        np.concatenate(train_labels),  # Объединение массивов вдоль существующей оси (горизонтально)
        np.vstack(test_features),
        np.concatenate(test_labels),
    )
