import numpy as np
import pytest
from data_analysis.preprocessing import get_boxplot_outliers, train_test_split
from models.knn import euclidean_dist


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


def test_euclidean_dist():
    # Проверка 2D
    assert np.allclose(
        euclidean_dist(np.array([[0,0]]), np.array([[3,4]])), 
        [5.0]
    )
    # Проверка 3D
    assert np.allclose(
        euclidean_dist(np.array([[1,1,1]]), np.array([[4,5,6]])), 
        [np.sqrt(9+16+25)]
    )