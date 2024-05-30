import numpy as np
from typing import Callable, List, Union

def k_distance(distances: List[np.ndarray], neighbours: List[np.ndarray]) -> List[float]:
    return [distance_row[-1] for distance_row in distances]

def mean_knn_distance(distances: List[np.ndarray], neighbours: List[np.ndarray]) -> List[float]:
    return [distance_row.mean() for distance_row in distances]

outlier_factor_dict = {'k_distance': k_distance, 'mean_knn_distance': mean_knn_distance}