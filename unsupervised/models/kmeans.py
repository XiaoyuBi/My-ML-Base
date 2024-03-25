import numpy as np
from typing import List


class MyKMeans:
    """
    Simple implementation of K-Means clustering algorithm.
    Assume random centroid initialization.
    Reference: 
    https://github.com/eriklindernoren/ML-From-Scratch/blob/master/mlfromscratch/unsupervised_learning/k_means.py
    """

    def __init__(self, 
            k: int = 5,
            max_iteration: int = 100,
            tol: float = 1e-4
        ) -> None:
        self.k = k 
        self.max_iteration = max_iteration
        self.tol = tol
    
    def _init_centroids(self, X: np.array):
        """
        Input: X (n_samples, n_features)
        Output: centroids (k, n_features)
        """
        n_samples, n_features = np.shape(X)
        self.centroids = np.zeros((self.k, n_features))

        # pick centroids from samples
        for i in range(self.k):
            sample = X[np.random.choice(range(n_samples))]
            self.centroids[i] = sample
    
    def _closest_centroid(self, X) -> np.array:
        """
        Input: X (n_samples, n_features) 
               self.centroids (k, n_features)
        Output: closest_idx (n_samples, ) with values in range [0, k-1]
        """
        X = np.expand_dims(X, axis = 1) # (n_samples, 1, n_features)
        dists = np.linalg.norm(X - self.centroids, axis = 2)
        closest_idx = np.argmin(dists, axis = 1)

        return closest_idx
    
    def _recenter(self, X):
        """
        Input: X (n_samples, n_features) 
               self.centroids (k, n_features)
        Output: new_centroids (k, n_features)
        """
        closest_idx = self._closest_centroid(X)
        new_centroids = np.zeros_like(self.centroids)
        cluster_counts = [0] * self.k

        for i, cluster_id in enumerate(closest_idx):
            new_centroids[cluster_id] += X[i]
            cluster_counts[cluster_id] += 1
        
        # Here, we can assure that every cluster_count is non-zero,
        # because we initialize centroids with sample data points.
        # If we use other initialization, there is no guarantee.
        for cluster_id in range(self.k):
            new_centroids[cluster_id] /= cluster_counts[cluster_id]
        
        self.centroids = new_centroids
    
    def fit(self, X):
        self._init_centroids(X)

        for _ in range(self.max_iteration):
            prev_centroids = self.centroids
            self._recenter(X)

            if np.linalg.norm(prev_centroids - self.centroids) < self.tol:
                print(f"Early stopping for convergence at iteration {_}.")
                break
    
    def predict(self, X):
        return self._closest_centroid(X)
