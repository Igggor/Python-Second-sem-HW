from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
from typing import Union
from models.knn import KNearestNeighbors, WeightedKNearestNeighbors


class AnimationKNN:
    """Класс для создания анимации работы KNN"""
    def __init__(self):
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.scatter = None
        self.neighbors_lines = []
        self.prediction_text = None

    def create_animation(
        self,
        knn: Union[WeightedKNearestNeighbors, KNearestNeighbors],
        X_test: np.ndarray,
        y_test: np.ndarray,
        path_to_save: str = "",
    ) -> FuncAnimation:
        """Создание анимации работы алгоритма"""
        predictions = knn.predict(X_test)

        def update(frame):
            self.ax.clear()
            x = X_test[frame]

            # Отображаем обучающие данные
            self.ax.scatter(
                knn.X_train[:, 0], knn.X_train[:, 1],
                c=knn.y_train, alpha=0.3, label="Train data"
            )

            # Вычисляем ближайших соседей
            distances = knn.calc_distances(knn.X_train, x)
            nearest_indices = np.argsort(distances)[:knn.n_neighbors]

            # Отображаем тестовую точку
            self.ax.scatter(
                x[0], x[1],
                c='red' if predictions[frame] == y_test[frame] else 'black',
                marker='x', s=100, label="Test point"
            )

            # Рисуем линии к соседям
            for idx in nearest_indices:
                neighbor = knn.X_train[idx]
                self.ax.plot(
                    [x[0], neighbor[0]], [x[1], neighbor[1]],
                    'gray', linestyle='--', alpha=0.5
                )

            # Добавляем информацию о предсказании
            self.ax.set_title(
                f"Sample {frame+1}/{len(X_test)}\n"
                f"True: {y_test[frame]}, Predicted: {predictions[frame]}\n"
                f"Correct: {predictions[frame] == y_test[frame]}"
            )
            self.ax.legend()

        anim = FuncAnimation(
            self.fig, update, frames=len(X_test),
            interval=500, repeat=False
        )

        if path_to_save:
            anim.save(path_to_save, writer='pillow', fps=2)

        return anim
