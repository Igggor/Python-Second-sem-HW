from typing import Callable
import numpy as np


def euclidean_dist(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """Вычисляет евклидово расстояние между точками или наборами точек."""
    x1 = np.atleast_2d(x1)
    x2 = np.atleast_2d(x2)

    # Проверка совместимости размерностей
    if x1.shape[1] != x2.shape[1]:
        raise ValueError(f"Несовпадение размерностей признаков: {x1.shape[1]} и {x2.shape[1]}")

    # Для случая, когда x2 - одна точка
    if len(x2) == 1:
        return np.sqrt(np.sum((x1 - x2) ** 2, axis=1))

    # Для общего случая
    return np.sqrt(np.sum((x1[:, np.newaxis, :] - x2[np.newaxis, :, :]) ** 2, axis=2)).squeeze()


class KNearestNeighbors:
    """Классический алгоритм K ближайших соседей"""
    def __init__(self, n_neighbors: int = 5, calc_distances: Callable = euclidean_dist):
        self.n_neighbors = n_neighbors
        self.calc_distances = calc_distances
        self.X_train = None
        self.y_train = None
        self._fitted = False

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """Обучение модели.

        Args:
            X_train: Массив признаков формы (n_samples, n_features)
            y_train: Массив меток формы (n_samples,)
        """
        if len(X_train) != len(y_train):
            raise ValueError("X_train и y_train должны иметь одинаковую длину")

        self.X_train = X_train
        self.y_train = y_train
        self._fitted = True

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Предсказание меток для тестовых данных.

        Args:
            X_test: Массив признаков формы (n_samples, n_features)

        Returns:
            Массив предсказанных меток формы (n_samples,)
        """
        if not self._fitted:
            raise ValueError("Сначала нужно вызвать fit()")
        if self.n_neighbors > len(self.X_train):
            raise ValueError(
                f"n_neighbors ({self.n_neighbors}) > числа образцов ({len(self.X_train)})"
                )

        X_test = np.atleast_2d(X_test)
        distances = self.calc_distances(self.X_train, X_test)

        # Для случая, когда X_test - одна точка
        if distances.ndim == 1:
            nearest_indices = np.argpartition(distances, self.n_neighbors)[:self.n_neighbors]
            nearest_labels = self.y_train[nearest_indices]
            return self._majority_vote(nearest_labels)
        else:
            nearest_indices = np.argpartition(distances, self.n_neighbors, axis=1)[
                :, :self.n_neighbors
            ]
            nearest_labels = self.y_train[nearest_indices]
            return np.array([self._majority_vote(row) for row in nearest_labels])

    def _majority_vote(self, labels: np.ndarray) -> int:
        """Вспомогательная функция для голосования большинством."""
        vals, counts = np.unique(labels, return_counts=True)
        return vals[np.argmax(counts)]

    def score(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        """Оценка точности модели.

        Args:
            X_test: Массив признаков формы (n_samples, n_features)
            y_test: Массив истинных меток формы (n_samples,)

        Returns:
            Точность предсказаний (0-1)
        """
        preds = self.predict(X_test)
        return np.mean(preds == y_test)


class WeightedKNearestNeighbors:
    """Взвешенный алгоритм K ближайших соседей с ядром Епанечникова"""
    def __init__(self, n_neighbors: int = 5, calc_distances: Callable = euclidean_dist):
        self.n_neighbors = n_neighbors
        self.calc_distances = calc_distances
        self.X_train = None
        self.y_train = None
        self._fitted = False

    @staticmethod
    def epanechnikov_kernel(u: np.ndarray) -> np.ndarray:
        """Ядро Епанечникова"""
        return np.where(np.abs(u) <= 1, 0.75 * (1 - u**2), 0)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        if len(X_train) != len(y_train):
            raise ValueError("X_train и y_train должны иметь одинаковую длину")

        self.X_train = X_train
        self.y_train = y_train
        self._fitted = True

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Предсказание меток для тестовых данных с учетом весов.

        Args:
            X_test: Массив признаков формы (n_samples, n_features)

        Returns:
            Массив предсказанных меток формы (n_samples,)
        """
        if not self._fitted:
            raise ValueError("Сначала нужно вызвать fit()")
        if self.n_neighbors > len(self.X_train):
            raise ValueError(
                f"n_neighbors ({self.n_neighbors}) > числа образцов ({len(self.X_train)})"
            )

        predictions = []
        distances = self.calc_distances(self.X_train, X_test)

        for i in range(len(X_test)):
            point_distances = distances[i] if distances.ndim > 1 else distances
            nearest_indices = np.argpartition(point_distances, self.n_neighbors)[:self.n_neighbors]

            h = max(point_distances[nearest_indices[-1]], 1e-10)  # ширина окна
            # + еще сделал защиту от 0 чтобы потом не делить на 0
            normalized_dists = point_distances[nearest_indices] / h
            weights = self.epanechnikov_kernel(normalized_dists)

            nearest_labels = self.y_train[nearest_indices]
            unique_classes = np.unique(self.y_train)
            class_scores = {cls: 0 for cls in unique_classes}

            for cls, w in zip(nearest_labels, weights):
                class_scores[cls] += w

            predictions.append(max(class_scores.items(), key=lambda x: x[1])[0])

        return np.array(predictions)
