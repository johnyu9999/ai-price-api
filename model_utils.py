import os
import joblib

def load_model(version: str):
    model_path = f"model_registry/model_{version}.pkl"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found for version '{version}': {model_path}")

    model = joblib.load(model_path)
    return model
