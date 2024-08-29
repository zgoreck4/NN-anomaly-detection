# Unsupervised anomaly detection based on dissimilarity to neighbours

## Description

The project includes the implementation of an unsupervised anomaly detection algorithm based on dissimilarity to neighbors, along with a custom implementation of the k-nearest neighbors (kNN) algorithm.

## Example usage
```python
from algorithm.NNAnomalyDetector import NNAnomalyDetector
detector = NNAnomalyDetector(k=k, metric=metric, outlier_factor_input='lof')
detector.fit(X_train)
factor = detector.predict(X_test)
```