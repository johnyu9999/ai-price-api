# train.py
import argparse
import joblib
import os
import numpy as np
from sklearn.linear_model import LinearRegression

def train_and_save(version: str):
    np.random.seed(42 if version == "v1" else 123)
    X = 2 * np.random.rand(100, 1)
    y = 4 + 3 * X + np.random.randn(100, 1)

    model = LinearRegression()
    model.fit(X, y)

    path = f"model_registry/model_{version}.pkl"
    joblib.dump(model, path)
    print(f"✅ Model saved to {path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True, help="Model version name, e.g. v2")
    args = parser.parse_args()

    train_and_save(args.version)
