import matplotlib.pyplot as plt
from sklearn import datasets as skd
from data_analysis.visualization import visualize_distribution, AxisNames, DiagramTypes
from data_analysis.preprocessing import get_boxplot_outliers, train_test_split
from models.knn import KNearestNeighbors, WeightedKNearestNeighbors
from models.metrics import accuracy
from utils.animation import AnimationKNN


def main():
    # Загрузка данных
    points, labels = skd.make_moons(n_samples=400, noise=0.3)

    # Визуализация распределения фруктов
    visualize_distribution(
        points,
        diagram_type=[DiagramTypes.Violin, DiagramTypes.Hist, DiagramTypes.Boxplot],
        diagram_axis=[AxisNames.X, AxisNames.Y],
        path_to_save="fruit_distributions.png"
    )

    # Поиск аномальных точек падения (Выбросов)
    outliers = get_boxplot_outliers(points)
    print(f"Найдено {len(outliers)} аномальных точек падения")

    # Разделение данных
    X_train, y_train, X_test, y_test = train_test_split(
        points, labels, train_ratio=0.7, shuffle=True
    )

    # Обучение и оценка качества моделей
    models = {
        "Обычный KNN": KNearestNeighbors(n_neighbors=5),
        "Взвешенный KNN": WeightedKNearestNeighbors(n_neighbors=10)
    }
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        acc = accuracy(y_test, pred)
        print(f"{name} точность: {acc:.2f}")

    # Визуализация работы алгоритма
    animator = AnimationKNN()
    animator.create_animation(
        models["Обычный KNN"], 
        X_test[:20], 
        y_test[:20],
        path_to_save="fruit_classification.gif"
    )
    plt.show()


if __name__ == "__main__":
    main()
