import numpy as np
from sklearn.metrics.pairwise import distance_metrics
from typing import Callable, List, Union
from KNN import KNN
from utils import outlier_factor_dict

class NNAnomalyDetector:
    def __init__(self, k: int, metric: str | Callable, outlier_factor_input: str | Callable):
        self.k = k
        self.metric = metric

        outlier_factor = outlier_factor_dict.get(outlier_factor_input)
        if outlier_factor==None:
            raise ValueError("Invalid outlier factor metric name")
        self.outlier_factor = outlier_factor

        self.fitted = False

    def train(self, X: np.ndarray) -> None:
        self.kNN = KNN(self.k, self.metric)
        self.kNN.fit(X)
        self.fitted = True
        print(self.fitted)

    def _k_distance(self, distances: List[np.ndarray], neighbours: List[np.ndarray]) -> List[float]:
        return [distance_row[-1] for distance_row in distances]
    
    def _mean_knn_distance(self, distances: List[np.ndarray], neighbours: List[np.ndarray]) -> List[float]:
        return [distance_row.mean() for distance_row in distances]
    
    def predict(self, X: np.ndarray, thresh: float = None) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        if not self.fitted:
            raise ValueError("Model has not been trained yet")
        
        distances, indexes = self.kNN.predict(X)
        outlier_factor_list = self.outlier_factor(distances, indexes)
        
        if thresh is not None:
            return outlier_factor_list, (np.array(outlier_factor_list) > thresh).astype(int)
        else:
            return outlier_factor_list