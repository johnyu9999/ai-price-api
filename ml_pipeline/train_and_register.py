# ml_pipeline/train_and_register.py

import os
import joblib
import argparse
import numpy as np
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from pathlib import Path
from data_generator import generate_linear_data

MODEL_DIR = Path(__file__).parent / "model_store"
META_PATH = Path(__file__).parent / "model_registry.jsonl"
MODEL_DIR.mkdir(exist_ok=True)

def validate_model(model, X, y, version):
    try:
        y_pred = model.predict(X)
        mse = mean_squared_error(y, y_pred)
        print(f"✅ Validation passed | MSE={mse:.4f}")
        return True, mse
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        return False, None

def save_metadata(version, n_samples, n_features, noise, mse):
    meta = {
        "version": version,
        "samples": n_samples,
        "features": n_features,
        "noise": noise,
        "mse": mse,
        "timestamp": datetime.utcnow().isoformat()
    }
    with open(META_PATH, "a") as f:
        f.write(f"{meta}\n")

def main(args):
    X, y, weights = generate_linear_data(
        n_samples=args.samples,
        n_features=args.features,
        noise=args.noise,
        seed=args.seed
    )

    model = LinearRegression()
    model.fit(X, y)
    model.fit(X, y)
    _ = model.predict(X[:5])  # 让 model 的 predict 机制完整初始化

    # Validate before saving
    is_valid, mse = validate_model(model, X, y, args.version)
    if not is_valid:
        print("🛑 Model validation failed. Not saving.")
        return

    model_path = MODEL_DIR / f"{args.version}.pkl"
    joblib.dump(model, model_path)
    print(f"✅ Model saved to {model_path}")

    save_metadata(args.version, args.samples, args.features, args.noise, mse)
    print(f"📝 Metadata logged to {META_PATH}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--features", type=int, default=1)
    parser.add_argument("--noise", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    main(args)
