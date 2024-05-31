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
        self.pred_distances = None
        self.pred_neighbours = None

    def _k_distance(self) -> List[float]:
        return [distance_row[-1] for distance_row in self.pred_distances]
    
    def _mean_knn_distance(self) -> List[float]:
        return [distance_row.mean() for distance_row in self.pred_distances]
    
    def _loc_reachability_density(self) -> List[float]:
        loc_reachability_density_list = []
        for example_pred_distances, example_pred_neighbours in zip(self.pred_distances, self.pred_neighbours):
            max_nn_dist = example_pred_distances[-1]
            reachability_sum = 0
            for distance, neighbour in zip(example_pred_distances, example_pred_neighbours):
                neigh_dist, _ =  self.kNN.predict(self.kNN.X_train[neighbour].reshape(1, -1))
                reachability_sum += max(max_nn_dist, neigh_dist[0][-1])
            loc_reachability_density_list.append(1/(reachability_sum/len(example_pred_neighbours)))
        return loc_reachability_density_list

    def _lof(self) -> List[float]:
        lof_list = []
        for example_pred_distances, example_pred_neighbours in zip(self.pred_distances, self.pred_neighbours):
            print(f"example_self.pred_distances: {example_pred_distances}")
            sum = 0
            for distance, neighbour in zip(example_pred_distances, example_pred_neighbours):
                neigh_dist, _ =  self.kNN.predict(self.kNN.X_train[neighbour].reshape(1, -1))
                print(f"neigh_dist: {neigh_dist}")
                sum += neigh_dist[0][-1]/example_pred_distances[-1]
                print(f"sum: {sum}")
            lof_list.append(sum/len(example_pred_neighbours))
        return lof_list

    def train(self, X: np.ndarray) -> None:
        self.kNN = KNN(self.k, self.metric)
        self.kNN.fit(X)
        self.fitted = True
        print(self.fitted)
    
    def predict(self, X: np.ndarray, thresh: float = None) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        if not self.fitted:
            raise ValueError("Model has not been trained yet")
        
        self.pred_distances, self.pred_neighbours = self.kNN.predict(X)
        outlier_factor_list = self.outlier_factor()
        if thresh is not None:
            return outlier_factor_list, (np.array(outlier_factor_list) > thresh).astype(int)
        else:
            return outlier_factor_list