import numpy as np
from sklearn.metrics.pairwise import distance_metrics
from typing import Callable, List, Union
from KNN import KNN

class NNAnomalyDetector:
    def __init__(self, k: int, metric: str | Callable, outlier_factor_input: str | Callable):
        self.k = k
        self.metric = metric

        outlier_factor_dict = {'k_distance': self._k_distance, 'mean_knn_distance': self._mean_knn_distance, "negative_loc_reachability_density": self._negative_loc_reachability_density, "lof": self._lof}
        outlier_factor = outlier_factor_dict.get(outlier_factor_input)
        if outlier_factor==None:
            raise ValueError("Invalid outlier factor metric name")
        self.outlier_factor = outlier_factor

        self.fitted = False

    def _k_distance(self, distances: np.ndarray, *argv) -> float:
        return distances[-1]
    
    def _mean_knn_distance(self, distances: np.ndarray, *argv) -> float:
        return distances.mean()
    
    def _reachability(self, distance: np.ndarray, neighbour_idx: int) -> float:
        neigh_dist = self.kNN.train_distances[neighbour_idx]
        return max(distance, self._k_distance(neigh_dist))
    
    def _negative_loc_reachability_density(self, distances: np.ndarray, neighbours_idx: np.ndarray) -> float:
        reachability_sum = 0
        for distance, neighbour_idx in zip(distances, neighbours_idx):
            reachability_sum += self._reachability(distance, neighbour_idx)
        return -1/(reachability_sum/len(neighbours_idx))

    def _lof(self, distances: np.ndarray, neighbours_idx: np.ndarray) -> float:
        sum = 0
        for distance, neighbour_idx in zip(distances, neighbours_idx):
            neigh_distances = self.kNN.train_distances[neighbour_idx]
            neigh_neighbours_idx = self.kNN.train_neigh_idx[neighbour_idx]
            sum += self._negative_loc_reachability_density(neigh_distances, neigh_neighbours_idx)
        return sum/self._negative_loc_reachability_density(distances, neighbours_idx)/len(neighbours_idx)

    def fit(self, X: np.ndarray) -> None:
        self.kNN = KNN(self.k, self.metric)
        self.kNN.fit(X)
        self.fitted = True
    
    def predict(self, X: np.ndarray, thresh: float = None) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        if not self.fitted:
            raise ValueError("Model has not been trained yet")
           
        distances, neighbours_idx = self.kNN.predict(X)

        outlier_factor_list = []
        for example_distances, example_neighbours_idx in zip(distances, neighbours_idx):
            example_outlier_factor = self.outlier_factor(example_distances, example_neighbours_idx)
            outlier_factor_list.append(example_outlier_factor)

        if thresh is not None:
            return outlier_factor_list, (np.array(outlier_factor_list) > thresh).astype(int)
        else:
            return outlier_factor_list