import matplotlib.pyplot as plt
from sklearn import datasets as skd
from data_analysis.visualization import visualize_distribution, AxisNames, DiagramTypes
from data_analysis.preprocessing import get_boxplot_outliers, train_test_split
from models.knn import KNearestNeighbors, WeightedKNearestNeighbors
from models.metrics import accuracy
from utils.animation import AnimationKNN


def main():
    # 1. Загрузка данных
    points, labels = skd.make_moons(n_samples=400, noise=0.3)

    # 2. Визуализация распределений
    visualize_distribution(
        points,
        diagram_type=[DiagramTypes.Violin, DiagramTypes.Hist, DiagramTypes.Boxplot],
        diagram_axis=[AxisNames.X, AxisNames.Y],
        path_to_save="distributions.png"
    )

    # 3. Поиск выбросов
    outliers = get_boxplot_outliers(points)
    print(f"Found {len(outliers)} outliers in points")

    # 4. Разделение данных
    X_train, y_train, X_test, y_test = train_test_split(
        points, labels, train_ratio=0.7, shuffle=True
    )

    # 5. Обучение и оценка моделей
    knn = KNearestNeighbors(n_neighbors=5)
    knn.fit(X_train, y_train)
    knn_pred = knn.predict(X_test)
    knn_acc = accuracy(y_test, knn_pred)
    print(f"KNN accuracy: {knn_acc:.2f}")

    weighted_knn = WeightedKNearestNeighbors(n_neighbors=10)
    weighted_knn.fit(X_train, y_train)
    weighted_pred = weighted_knn.predict(X_test)
    weighted_acc = accuracy(y_test, weighted_pred)
    print(f"Weighted KNN accuracy: {weighted_acc:.2f}")

    # 6. Визуализация работы алгоритма
    animator = AnimationKNN()
    animator.create_animation(
        knn, X_test[:20], y_test[:20], path_to_save="knn_animation.gif"
    )
    plt.show()


if __name__ == "__main__":
    main()
