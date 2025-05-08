from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
from typing import Union
from models.knn import KNearestNeighbors, WeightedKNearestNeighbors


class AnimationKNN:
    """Класс для создания анимации работы KNN с визуализацией области соседей"""
    def __init__(self):
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.scatter = None
        self.neighbors_lines = []
        self.neighbors_circle = None
        self.prediction_text = None

    def create_animation(
        self,
        knn: Union[WeightedKNearestNeighbors, KNearestNeighbors],
        X_test: np.ndarray,
        y_test: np.ndarray,
        path_to_save: str = "",
    ) -> FuncAnimation:
        """Создание анимации работы алгоритма с круговой областью соседей"""
        predictions = knn.predict(X_test)

        def update(frame):
            self.ax.clear()
            x = X_test[frame]
            is_correct = predictions[frame] == y_test[frame]

            # Отображаю обучающие данные
            self.ax.scatter(
                knn.X_train[:, 0], knn.X_train[:, 1],
                c=knn.y_train, alpha=0.3, label="Обучающие данные"
            )

            # Вычисляю ближайших соседей
            distances = knn.calc_distances(knn.X_train, x)
            nearest_indices = np.argsort(distances)[:knn.n_neighbors]
            max_distance = distances[nearest_indices[-1]]  # Радиус для круга

            # Рисую круг, охватывающий соседей
            circle = plt.Circle(
                (x[0], x[1]), max_distance,
                color='green', fill=False, linestyle='--', alpha=0.5,
                label=f'Область {knn.n_neighbors} соседей'
            )
            self.ax.add_patch(circle)

            point_color = 'green' if is_correct else 'red'
            self.ax.scatter(
                x[0], x[1],
                c=point_color, marker='X', s=150,
                label="Тестовая точка", edgecolors='black'
            )

            # Линии к соседям
            for idx in nearest_indices:
                neighbor = knn.X_train[idx]
                self.ax.plot(
                    [x[0], neighbor[0]], [x[1], neighbor[1]],
                    'gray', linestyle=':', alpha=0.7, linewidth=1
                )

            status = "Верно" if is_correct else "Ошибка"
            title_color = "green" if is_correct else "red"
            self.ax.set_title(
                f"Точка: ({x[0]:.2f}, {x[1]:.2f}) - {status}\n"
                f"Истинный класс: {y_test[frame]}, Предсказанный: {predictions[frame]}\n"
                f"К соседей: {knn.n_neighbors}, Радиус: {max_distance:.2f}",
                color=title_color, fontsize=12
            )

            self.ax.legend(loc='upper right')
            self.ax.grid(True, alpha=0.3)
            self.ax.set_xlabel("Признак 1")
            self.ax.set_ylabel("Признак 2")
            self.ax.set_aspect('equal')

        anim = FuncAnimation(
            self.fig, update, frames=len(X_test),
            interval=500, repeat=False
        )

        if path_to_save:
            anim.save(path_to_save, writer='pillow', fps=1, dpi=100)

        return anim
