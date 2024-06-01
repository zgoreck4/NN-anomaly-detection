# TODO fix imports
from ..algorithm.NNAnomalyDetector import NNAnomalyDetector
from sklearn.neighbors import LocalOutlierFactor
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ucimlrepo import fetch_ucirepo 
# fetch dataset 
wine = fetch_ucirepo(id=109) 
  
# data
# TODO preprocessing and save data (another file)
X = wine.data.features 
y = wine.data.targets 
X = np.array(X)
X_train, X_test = train_test_split(X, test_size=0.2) # TODO X_train should not contain anomalies

metric = 'euclidean'
k = 2

detector = NNAnomalyDetector(k=k, metric=metric, outlier_factor_input='lof')
detector.fit(X_train)
factor = detector.predict(X_test)
factor = np.round(factor, 5)

lof = LocalOutlierFactor(n_neighbors=k, metric=metric, novelty=True)
lof.fit(X_train)
factor_lof = -(lof.decision_function(X_test)+lof.offset_)
factor_lof = np.round(factor_lof, 5)

assert np.array_equal(factor, factor_lof), f"Local outlier factors aren't the same: {factor} and {factor_lof}"