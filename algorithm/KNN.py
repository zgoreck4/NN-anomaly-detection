import numpy as np
from sklearn.metrics.pairwise import distance_metrics
from typing import Callable, List, Tuple

class KNN:
    """
    A class that implements the k-Nearest Neighbors (kNN) algorithm.
    In the case where the k-th neighbor is as distant as the k+n-th neighbor the algorithm may return
    more than k nearest neighbors.

    Parameters:
    ----------
    k : int
        The number of nearest neighbors to consider.
    metric : str or Callable
        The distance metric to use. Can be either a string specifying one of the metrics from
        `sklearn.metrics.pairwise.distance_metrics()` or a custom callable function.

    Attributes:
    ----------
    k : int
        The number of nearest neighbors.
    metric : str or Callable
        The distance metric function.
    X_train : np.ndarray
        The training dataset.
    train_distances : List[np.ndarray]
        A list of distance vectors to the k nearest neighbors for each training data point.
    train_neigh_idx : List[np.ndarray]
        A list of index vectors representing the k nearest neighbors for each training data point.
    """

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
        """
        Computes the distances from a given point `x1` to all points in the training set.

        Parameters:
        ----------
        x1 : np.ndarray
            The point from which distances are calculated.
        exclude_index : int, optional
            The index of a point in `X_train` to exclude from the distance calculation. Useful during training.

        Returns:
        ----------
        List[Tuple[float, int]]
            A list of tuples containing the distance and the index of the corresponding training point.
        """
        dist_idx = []
        for j, x2 in enumerate(self.X_train):
            if exclude_index==j:
                continue
            dist_idx.append((self.metric(x1.reshape(1, -1), x2.reshape(1, -1))[0,0], j))
        return dist_idx
    
    def _choose_kNN(self, dist_idx: List[Tuple[float, int]]) -> Tuple[np.ndarray, np.ndarray]: # ->???
        """
        Selects the k nearest neighbors from a list of distances.

        Parameters:
        ----------
        dist_idx : List[Tuple[float, int]]
            A list of tuples containing distances and indices of points in the training set.

        Returns:
        ----------
        Tuple[np.ndarray, np.ndarray]
            A tuple containing two arrays: the distances of the k nearest neighbors and their indices.
        """
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
        """
        Fits the model to the training data and predicts the nearest neighbors for each training point.

        Parameters:
        ----------
        X : np.ndarray
            The training dataset.
        """
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
        """
        Trains the kNN model using the provided training dataset.

        Parameters:
        ----------
        X_train : np.ndarray
            The training dataset.
        """
        self.X_train = X_train
        self._fit_predict(X_train)
    
    def predict(self, X: np.ndarray) -> tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Predicts the k nearest neighbors for each point in the input data.

        Parameters:
        ----------
        X : np.ndarray
            The input data for which to predict the nearest neighbors.

        Returns:
        ----------
        tuple[List[np.ndarray], List[np.ndarray]]
            A tuple containing two lists:
            - `neighbours_dist_list`: A list of arrays, each containing the distances to the k nearest neighbors.
            - `neighbours_idx_list`: A list of arrays, each containing the indices of the k nearest neighbors.
        """
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
        """
        Trains the kNN model on the provided data and returns the k nearest neighbors for each point in the training data.

        Parameters:
        ----------
        X : np.ndarray
            The training dataset.

        Returns:
        ----------
        tuple[List[np.ndarray], List[np.ndarray]]
            A tuple containing two lists:
            - `neighbours_dist_list`: A list of arrays, each containing the distances to the k nearest neighbors.
            - `neighbours_idx_list`: A list of arrays, each containing the indices of the k nearest neighbors.
        """
        self.fit(X)
        return self.train_distances, self.train_neigh_idx