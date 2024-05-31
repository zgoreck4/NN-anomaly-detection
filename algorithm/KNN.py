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
        self.train_distances = None
        self.train_neigh_idx = None

    def _compute_distances(self, x1: np.ndarray, exclude_index: int=None) -> List[Tuple[float, int]]:
        dist_idx = []
        for j, x2 in enumerate(self.X_train):
            if exclude_index==j:
                continue
            dist_idx.append((self.metric(x1.reshape(1, -1), x2.reshape(1, -1))[0,0], j))
        return dist_idx
    
    def _choose_kNN(self, dist_idx: List[Tuple[float, int]]) -> Tuple[np.ndarray, np.ndarray]: # ->???
        dist_idx.sort(key=lambda x: x[0])
        neighbours_dist = []
        neighbours_idx = []
        kth_distance = dist_idx[self.k - 1][0]
        
        for distance, index in dist_idx:
            if distance <= kth_distance:
                neighbours_dist.append(distance)
                neighbours_idx.append(index)
            else:
                break
        return np.array(neighbours_dist), np.array(neighbours_idx)
    
    def _fit_predict(self, X: np.ndarray) -> None:
        neighbours_dist_list = []
        neighbours_idx_list = []
        for i, x in enumerate(X):
            dist_idx = self._compute_distances(x, exclude_index=i) # main difference between predict method
            neighbours_dist, neighbours_idx = self._choose_kNN(dist_idx)
            neighbours_dist_list.append(neighbours_dist)
            neighbours_idx_list.append(neighbours_idx)
        self.train_distances = neighbours_dist_list
        self.train_neigh_idx = neighbours_idx_list
        
    def fit(self, X_train: np.ndarray) -> None:
        self.X_train = X_train
        self._fit_predict(X_train)
    
    def predict(self, X: np.ndarray) -> tuple[List[np.ndarray], List[np.ndarray]]:
        if len(self.X_train)==0:
            raise ValueError("Model has not been trained yet")
        neighbours_dist_list = []
        neighbours_idx_list = []
        for x in X:
            dist_idx = self._compute_distances(x)
            neighbours_dist, neighbours_idx = self._choose_kNN(dist_idx)
            neighbours_dist_list.append(neighbours_dist)
            neighbours_idx_list.append(neighbours_idx)
        return neighbours_dist_list, neighbours_idx_list
    
    def fit_predict(self, X: np.ndarray) -> tuple[List[np.ndarray], List[np.ndarray]]:
        self.fit(X)
        return self.train_distances, self.train_neigh_idx