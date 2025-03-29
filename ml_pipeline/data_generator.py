import numpy as np

def generate_linear_data(n_samples=100, weights=[1.5, -2.0, 3.0], bias=4.2, noise_std=0.1):
    X = np.random.rand(n_samples, len(weights))
    y = X @ np.array(weights) + bias + np.random.normal(0, noise_std, size=n_samples)
    return X, y
