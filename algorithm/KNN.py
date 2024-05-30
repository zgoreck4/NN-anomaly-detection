import numpy as np
from sklearn.metrics.pairwise import distance_metrics
from typing import Callable, List, Tuple

class KNN:
    def __init__(self, k: int, metric: str | Callable):
        self.k = k
        if type(metric)==str:
            distance_metrics_dict = distance_metrics()
            if not metric in distance_metrics_dict.keys():
                raise ValueError("Invalid distance metric name")
            self.metric = distance_metrics_dict[metric]
        elif type(metric)==function:
            self.metric = metric
        else:
            raise TypeError("Invalid distance metric type")
        self.X_train = None

    def _compute_distances(self, x1: np.ndarray) -> List[Tuple[float, int]]:
        dist_idx = []
        for i, x2 in enumerate(self.X_train):
            dist_idx.append((self.metric(x1.reshape(1, -1), x2.reshape(1, -1))[0,0], i))
        return dist_idx
    
    def _choose_kNN(self, dist_idx: List[Tuple[float, int]]) -> Tuple[np.ndarray, np.ndarray]: # ->???
        dist_idx.sort(key=lambda x: x[0])
        neighbors_dist = []
        neighbors_idx = []
        kth_distance = dist_idx[self.k - 1][0]
        
        for distance, index in dist_idx:
            if distance <= kth_distance:
                neighbors_dist.append(distance)
                neighbors_idx.append(index)
            else:
                break
        return np.array(neighbors_dist), np.array(neighbors_idx)
        
    def fit(self, X_train: np.ndarray) -> None:
        self.X_train = X_train
    
    def predict(self, X: np.ndarray) -> tuple[List[np.ndarray], List[np.ndarray]]:
        if len(self.X_train)==0:
            raise ValueError("Model has not been trained yet")
        neighbors_dist_list = []
        neighbors_idx_list = []
        for x in X:
            dist_idx = self._compute_distances(x)
            neighbors_dist, neighbors_idx = self._choose_kNN(dist_idx)
            neighbors_dist_list.append(neighbors_dist)
            neighbors_idx_list.append(neighbors_idx)
        return neighbors_dist_list, neighbors_idx_list