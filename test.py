# # import pyteset
# import numpy as np
# from data_analysis.preprocessing import get_boxplot_outliers, train_test_split


# def check_get_boxplot_outliers():
#     # Тестовые данные с явными выбросами
#     test_data = np.array([
#         [1, 1.2],    # индекс 0
#         [2, 1.0],    # индекс 1
#         [3, 0.8],    # индекс 2
#         [4, -10.0],  # индекс 3 (выброс по Y)
#         [50, -0.5],  # индекс 4 (выброс по X)
#         [5, 100.0]   # индекс 5 (выброс по Y)
#     ])

#     outliers = get_boxplot_outliers(test_data)
#     assert len(outliers) == 3
#     print("Индексы выбросов:", outliers)  # Должно вернуть array([3, 4, 5])


# def check_train_test_split():
#     # Создадю тестовые данные с определенным сидом, чтобы можно было проверить.
#     np.random.seed(42)
#     features = np.random.rand(100, 2)  # 100 точек, 2 признака
#     targets = np.array([0]*70 + [1]*30)  # 70 класса 0 и 30 класса 1

#     train_feats, train_lbls, test_feats, test_lbls = train_test_split(
#         features, targets, train_ratio=0.7, shuffle=True, random_seed=42
#     )

#     # Результаты
#     print(f"Общее количество: {len(features)}")
#     print(f"Train размер: {len(train_feats)}")
#     print(f"Test размер: {len(test_feats)}")
#     print("\nРаспределение классов в исходных данных:")
#     print(f"Класс 0: {sum(targets == 0)}")
#     print(f"Класс 1: {sum(targets == 1)}")
#     print("\nРаспределение классов в train:")
#     print(f"Класс 0: {sum(train_lbls == 0)}")
#     print(f"Класс 1: {sum(train_lbls == 1)}")
#     print("\nРаспределение классов в test:")
#     print(f"Класс 0: {sum(test_lbls == 0)}")
#     print(f"Класс 1: {sum(test_lbls == 1)}")

# if __name__ == "__main__":
#     check_train_test_split()



import numpy as np
import pytest
from data_analysis.preprocessing import get_boxplot_outliers, train_test_split

# Фикстуры для тестовых данных
@pytest.fixture
def outlier_data():
    """Тестовые данные с явными выбросами"""
    return np.array([
        [1, 1.2],    # индекс 0
        [2, 1.0],    # индекс 1
        [3, 0.8],    # индекс 2
        [4, -10.0],  # индекс 3 (выброс по Y)
        [50, -0.5],  # индекс 4 (выброс по X)
        [5, 100.0]   # индекс 5 (выброс по Y)
    ])

@pytest.fixture
def classification_data():
    """Тестовые данные для проверки разделения"""
    np.random.seed(42)
    features = np.random.rand(100, 2)
    targets = np.array([0]*70 + [1]*30)
    return features, targets

class TestGetBoxplotOutliers:
    def test_outlier_detection(self, outlier_data):
        """Проверка обнаружения выбросов"""
        outliers = get_boxplot_outliers(outlier_data)
        assert len(outliers) == 3
        assert set(outliers) == {3, 4, 5}
    
    def test_no_outliers(self):
        """Проверка на данных без выбросов"""
        data = np.array([[1, 2], [2, 3], [3, 4]])
        outliers = get_boxplot_outliers(data)
        assert len(outliers) == 0

class TestTrainTestSplit:
    def test_split_sizes(self, classification_data):
        """Проверка размеров выборок"""
        features, targets = classification_data
        train_feats, _, test_feats, _ = train_test_split(
            features, targets, train_ratio=0.7
        )
        assert len(train_feats) == 70
        assert len(test_feats) == 30
    
    def test_class_balance(self, classification_data):
        """Проверка сохранения баланса классов"""
        features, targets = classification_data
        train_feats, train_lbls, test_feats, test_lbls = train_test_split(
            features, targets, train_ratio=0.7
        )
        
        # Проверяем пропорции в train
        assert np.isclose(
            sum(train_lbls == 0) / sum(train_lbls == 1),
            70 / 30,
            rtol=0.1
        )
        
        # Проверяем пропорции в test
        assert np.isclose(
            sum(test_lbls == 0) / sum(test_lbls == 1),
            70 / 30,
            rtol=0.1
        )
    
    def test_random_seed(self, classification_data):
        """Проверка воспроизводимости результатов"""
        features, targets = classification_data
        
        # Первый вызов
        train1, _, _, _ = train_test_split(
            features, targets, random_seed=42
        )
        
        # Второй вызов с тем же seed
        train2, _, _, _ = train_test_split(
            features, targets, random_seed=42
        )
        
        assert np.array_equal(train1, train2)

# Дополнительные проверки
def test_invalid_inputs():
    """Проверка обработки некорректных входных данных"""
    with pytest.raises(ValueError):
        train_test_split(np.array([[1, 2]]), np.array([1, 2]))  # Разная длина
        
    with pytest.raises(ValueError):
        train_test_split(np.array([[1, 2]]), np.array([1]), train_ratio=1.5)