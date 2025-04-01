# ml_pipeline/data_generator.py

import numpy as np

def generate_linear_data(n_samples=100, n_features=1, noise=0.1, seed=None):
    if seed:
        np.random.seed(seed)
    X = np.random.rand(n_samples, n_features)
    weights = np.random.randn(n_features)
    y = X @ weights + np.random.randn(n_samples) * noise + 4.2
    return X, y, weights
