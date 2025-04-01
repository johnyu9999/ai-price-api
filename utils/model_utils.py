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
    print(f"✅ model loaded: {model}")
    print(f"✅ coef_: {model.coef_}, intercept_: {model.intercept_}")
    print("✅ model.predict call (sanity):", model.predict([[0.1, 0.2, 0.3]]))
    return model
