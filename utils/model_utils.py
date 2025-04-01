import os
import joblib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # ../..
MODEL_STORE_DIR = PROJECT_ROOT / "ml_pipeline" / "model_store"

def load_model(version: str):
    model_path =  MODEL_STORE_DIR / f"model_{version}.pkl"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found for version '{version}': {model_path}")

    model = joblib.load(model_path)
    return model
