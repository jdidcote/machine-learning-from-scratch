from os import stat
from typing import Callable
from learners.base import BaseLearner

import numpy as np

from cost.least_squares import LeastSquaresCost
from learners.base import BaseLearner


class KMeansClustering(BaseLearner):
    def __init__(
        self, 
        K: int, 
        n_iter: int = 10, 
        n_runs: int = 10, 
        *args, 
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.K = K
        self.n_iter = n_iter
        self.n_runs = n_runs
        self.clusters = None
        self.centroids = None
        
    @staticmethod
    def distortion(x: np.ndarray, u: np.ndarray) -> float:
        """ Calculate distortion
        """
        return (np.sum(x - u)**2)
    
    @staticmethod
    def initialise_centroids(X: np.ndarray, K: int) -> np.array:
        """ Randomly initialise the cluster centroids
        """
        centroids_idxs = np.random.choice(np.arange(X.shape[0]), size=K, replace=False)
        return X[centroids_idxs, :]
    
    @staticmethod
    def get_closest_centroid(X: np.ndarray, centroids: np.ndarray, cost_func: Callable):
        """ For each data point return the closest centroid
        """
        clusters = []
        for x in X:
            costs = [cost_func(x, u) for u in centroids]
            clusters.append(np.argmin(costs))
        return np.array(clusters)
    
    @staticmethod
    def get_new_centroids(X: np.ndarray, clusters: np.ndarray):
        new_centroids = []
        for cluster in np.unique(clusters):
            new_centroids.append(X[(clusters == cluster)].mean(axis=0))
        return np.array(new_centroids)
    
    def k_means(self, X: np.ndarray, K: int, n_iter: np.ndarray):
        """ Run the k means clustering algorithm
        """
        centroids = self.initialise_centroids(X, K)
        for _ in range(n_iter):
            clusters = self.get_closest_centroid(X, centroids, self.distortion)
            centroids = self.get_new_centroids(X, clusters)
        return clusters, centroids
    
    def k_means_best(self, X: np.ndarray, K: int, n_runs: int, n_iter: int):
        results = []
        for _ in range(n_runs):
            clusters, centroids = self.k_means(X, K, n_iter)
            # Total distortion cost for algorithm
            dist = sum([self.distortion(X, centroid) for centroid in centroids])
            results.append((dist, clusters, centroids))
        
        self.cluster = sorted(results, key=lambda x: x[0])[0][1]
        self.centroids = sorted(results, key=lambda x: x[0])[0][2]

    def learn(self, X, y=None, padded=False):
        self.k_means_best(X=X, K=self.K, n_runs=self.n_runs, n_iter=self.n_iter)

    def predict():
        pass
    
    def predict_adhoc():
        pass
    