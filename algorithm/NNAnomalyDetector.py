import numpy as np
from sklearn.metrics.pairwise import distance_metrics
from typing import Callable, List, Union
from KNN import KNN

class NNAnomalyDetector:
    """
    A class for detecting anomalies using k-Nearest Neighbors (kNN) based methods.

    Parameters:
    ----------
    k : int
        The number of nearest neighbors to consider.
    metric : str or Callable
        The distance metric to use. Can be either a string specifying one of the metrics from
        `sklearn.metrics.pairwise.distance_metrics()` or a custom callable function.
    outlier_factor_input : str or Callable
        The method used to calculate the outlier factor. Can be either a custom function or one of the
        following predefined methods:
        - 'k_distance': Outlier factor based on the distance to the k-th nearest neighbor.
        - 'mean_knn_distance': Outlier factor based on the mean distance to the k nearest neighbors.
        - 'negative_loc_reachability_density': Negative local reachability density.
        - 'lof': Local Outlier Factor (LOF).
    
    Attributes:
    ----------
    k : int
        The number of nearest neighbors.
    metric : Callable
        The distance metric function.
    outlier_factor : Callable
        The function used to calculate the outlier factor.
    fitted : bool
        Indicates whether the model has been trained.
    kNN : KNN
        The trained kNN model.
    """

    def __init__(self, k: int, metric: str | Callable, outlier_factor_input: str | Callable):
        self.k = k
        self.metric = metric

        # Dictionary mapping outlier factor names to methods
        outlier_factor_dict = {
            'k_distance': self._k_distance,
            'mean_knn_distance': self._mean_knn_distance,
            "negative_loc_reachability_density": self._negative_loc_reachability_density,
            "lof": self._lof
        }
        outlier_factor = outlier_factor_dict.get(outlier_factor_input)
        if outlier_factor==None:
            raise ValueError("Invalid outlier factor metric name")
        self.outlier_factor = outlier_factor

        self.fitted = False

    def _k_distance(self, distances: np.ndarray, *argv) -> float:
        """
        Calculates the k-distance, which is the distance to the k-th nearest neighbor.

        Parameters:
        ----------
        distances : np.ndarray
            Array of distances to neighbors.

        Returns:
        ----------
        float
            The distance to the k-th nearest neighbor.
        """
        return distances[-1]
    
    def _mean_knn_distance(self, distances: np.ndarray, *argv) -> float:
        """
        Calculates the mean distance to the k nearest neighbors.

        Parameters:
        ----------
        distances : np.ndarray
            Array of distances to neighbors.

        Returns:
        ----------
        float
            The mean distance to the k nearest neighbors.
        """
        return distances.mean()
    
    def _reachability(self, distance: np.ndarray, neighbour_idx: int) -> float:
        """
        Calculates the reachability distance between a point and its neighbor.

        Parameters:
        ----------
        distance : np.ndarray
            The distance to the neighbor.
        neighbour_idx : int
            The index of the neighbor.

        Returns:
        ----------
        float
            The reachability distance, which is the maximum of the distance to the neighbor and its k-distance.
        """
        neigh_dist = self.kNN.train_distances[neighbour_idx]
        return max(distance, self._k_distance(neigh_dist))
    
    def _negative_loc_reachability_density(self, distances: np.ndarray, neighbours_idx: np.ndarray) -> float:
        """
        Calculates the negative local reachability density (LRD).

        Parameters:
        ----------
        distances : np.ndarray
            Array of distances to neighbors.
        neighbours_idx : np.ndarray
            Array of neighbor indices.

        Returns:
        ----------
        float
            The negative local reachability density.
        """
        reachability_sum = 0
        for distance, neighbour_idx in zip(distances, neighbours_idx):
            reachability_sum += self._reachability(distance, neighbour_idx)
        return -1/(reachability_sum/len(neighbours_idx))

    def _lof(self, distances: np.ndarray, neighbours_idx: np.ndarray) -> float:
        """
        Calculates the Local Outlier Factor (LOF) for a point.

        Parameters:
        ----------
        distances : np.ndarray
            Array of distances to neighbors.
        neighbours_idx : np.ndarray
            Array of neighbor indices.

        Returns:
        ----------
        float
            The Local Outlier Factor.
        """
        sum_lof = 0
        for distance, neighbour_idx in zip(distances, neighbours_idx):
            neigh_distances = self.kNN.train_distances[neighbour_idx]
            neigh_neighbours_idx = self.kNN.train_neigh_idx[neighbour_idx]
            sum_lof += self._negative_loc_reachability_density(neigh_distances, neigh_neighbours_idx)
        return sum_lof/self._negative_loc_reachability_density(distances, neighbours_idx)/len(neighbours_idx)

    def fit(self, X: np.ndarray) -> None:
        """
        Trains the NNAnomalyDetector model on the provided dataset.

        Parameters:
        ----------
        X : np.ndarray
            The training dataset.
        """
        self.kNN = KNN(self.k, self.metric)
        self.kNN.fit(X)
        self.fitted = True
    
    def predict(self, X: np.ndarray, thresh: float = None) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """
        Predicts anomalies in the given dataset.

        Parameters:
        ----------
        X : np.ndarray
            The dataset to classify.
        thresh : float, optional
            A threshold value for classifying a point as an anomaly. If provided, the method returns
            both the outlier factor and the binary classification (1 for anomaly, 0 for normal).
            If not provided, only the outlier factor is returned.

        Returns:
        ----------
        Union[np.ndarray, tuple[np.ndarray, np.ndarray]]
            - If `thresh` is None: Returns an array of outlier factors.
            - If `thresh` is provided: Returns a tuple where the first element is the array of outlier factors,
              and the second element is a binary array indicating whether each point is an anomaly.
        """
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