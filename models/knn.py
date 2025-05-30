from typing import Callable
import numpy as np


def euclidean_dist(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    if x1.ndim == 1:
        x1 = x1[np.newaxis, :]
    if x2.ndim == 1:
        x2 = x2[np.newaxis, :]

    # axis=1 для того, чтобы суммировать внутри каждой строки
    return np.sqrt(np.sum((x1 - x2) ** 2, axis=1))


class KNearestNeighbors:
    """Классический алгоритм K ближайших соседей"""
    def __init__(self, n_neighbors: int = 5, calc_distances: Callable = euclidean_dist):
        self.n_neighbors = n_neighbors
        self.calc_distances = calc_distances
        self.X_train = None
        self.y_train = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        predictions = []
        for x in X_test:
            distances = self.calc_distances(self.X_train, x)
            nearest_indices = np.argsort(distances)[:self.n_neighbors]
            """
            Нюансы:
                np.argsort(distances) возвращает индексы точек в порядке возрастания расстояния.
            """
            nearest_labels = self.y_train[nearest_indices]
            unique, counts = np.unique(nearest_labels, return_counts=True)
            predictions.append(unique[np.argmax(counts)])
        return np.array(predictions)


class WeightedKNearestNeighbors:
    """Взвешенный алгоритм K ближайших соседей"""
    def __init__(self, n_neighbors: int = 5, calc_distances: Callable = euclidean_dist):
        self.n_neighbors = n_neighbors
        self.calc_distances = calc_distances
        self.X_train = None
        self.y_train = None

    @staticmethod
    def epanechnikov_kernel(u: np.ndarray) -> np.ndarray:
        return np.where(np.abs(u) <= 1, 0.75 * (1 - u**2), 0)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        predictions = []
        for x in X_test:
            distances = self.calc_distances(self.X_train, x)
            nearest_indices = np.argsort(distances)[:self.n_neighbors]
            h = max(distances[nearest_indices[-1]], 10e-9)  # ширина окна
            # А еще добавил защиту от деления на нуль
            weights = self.epanechnikov_kernel(distances[nearest_indices] / h)
            nearest_labels = self.y_train[nearest_indices]

            unique = np.unique(nearest_labels)
            class_scores = {cls: 0.0 for cls in unique}

            for label, weight in zip(nearest_labels, weights):
                class_scores[label] += weight
            predictions.append(max(class_scores, key=class_scores.get))
        return np.array(predictions)
