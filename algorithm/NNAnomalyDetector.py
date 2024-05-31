import numpy as np
from sklearn.metrics.pairwise import distance_metrics
from typing import Callable, List, Union
from KNN import KNN

class NNAnomalyDetector:
    def __init__(self, k: int, metric: str | Callable, outlier_factor_input: str | Callable):
        self.k = k
        self.metric = metric

        outlier_factor_dict = {'k_distance': self._k_distance, 'mean_knn_distance': self._mean_knn_distance, "loc_reachability_density": self._loc_reachability_density, "lof": self._lof}
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
        neigh_dist, _ =  self.kNN.predict(self.kNN.X_train[neighbour_idx].reshape(1, -1))
        return max(distance, self._k_distance(neigh_dist[0]))
    
    def _loc_reachability_density(self, distances: np.ndarray, neighbours_idx: np.ndarray) -> float:
        reachability_sum = 0
        for distance, neighbour_idx in zip(distances, neighbours_idx):
            reachability_sum += self._reachability(distance, neighbour_idx)
        return 1/(reachability_sum/len(neighbours_idx))

    def _lof(self, distances: np.ndarray, neighbours_idx: np.ndarray) -> float:
        sum = 0
        for distance, neighbour_idx in zip(distances, neighbours_idx):
            neigh_distances, neigh_neighbours_idx =  self.kNN.predict(self.kNN.X_train[neighbour_idx].reshape(1, -1))
            print(f"neigh_distances: {neigh_distances}")
            sum += self._loc_reachability_density(neigh_distances[0], neigh_neighbours_idx[0])
            print(f"sum: {sum}")
        return sum/self._loc_reachability_density(distances, neighbours_idx)/len(neighbours_idx)

    def fit(self, X: np.ndarray) -> None:
        self.kNN = KNN(self.k, self.metric)
        self.kNN.fit(X)
        self.fitted = True
        print(self.fitted)
    
    def predict(self, X: np.ndarray, thresh: float = None) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        if not self.fitted:
            raise ValueError("Model has not been trained yet")
           
        distances, neighbours_idx = self.kNN.predict(X)
        print('distances')
        print(distances)
        print('neighbours_idx')
        print(neighbours_idx)

        outlier_factor_list = []
        for example_distances, example_neighbours_idx in zip(distances, neighbours_idx):
            example_outlier_factor = self.outlier_factor(example_distances, example_neighbours_idx)
            outlier_factor_list.append(example_outlier_factor)

        if thresh is not None:
            return outlier_factor_list, (np.array(outlier_factor_list) > thresh).astype(int)
        else:
            return outlier_factor_list