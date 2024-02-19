from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
from scipy.spatial.distance import cdist

class kNN(BaseEstimator, ClassifierMixin):
    def __init__(self, n_neighbors: int = 3):
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        self.X = X
        self.y = y.to_numpy()
        return self

    def predict(self, X):
        distances = cdist(X, self.X)
        distance_idx = np.argpartition(distances, self.n_neighbors, -1)[:, :self.n_neighbors]
        return np.sign(np.sum(self.y[distance_idx], axis=-1))