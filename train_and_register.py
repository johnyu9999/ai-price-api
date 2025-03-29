from sklearn.linear_model import LinearRegression
import joblib
import os
from datetime import datetime
from data_generator import generate_linear_data

import argparse
import yaml
import sys

yaml_path = sys.argv[1] if len(sys.argv) > 1 else "ml_pipeline/config.yaml"
with open(yaml_path, "r") as f:
    config = yaml.safe_load(f)

X, y = generate_linear_data(
    n_samples=config["samples"],
    weights=config["weights"],
    bias=config["bias"],
    noise_std=config["noise"]
)

version = config["version"]
note = config.get("note", "")

model = LinearRegression()
model.fit(X, y)

model_path = os.path.join("model_store", f"model_{version}.pkl")
joblib.dump(model, model_path)
print(f"✅ Model {version} saved to {model_path}")

from sklearn.metrics import mean_squared_error

y_pred = model.predict(X)
mse = mean_squared_error(y, y_pred)
print(f"📊 Model evaluation: MSE = {mse:.4f}")

import json

metadata_entry = {
    "version": version,
    "timestamp": datetime.utcnow().isoformat(),
    "mse": round(mse, 4),
    "weights": model.coef_.tolist(),
    "bias": model.intercept_.tolist() if hasattr(model.intercept_, 'tolist') else model.intercept_,
    "note": note
}

with open("model_store/metadata.jsonl", "a") as f:
    f.write(json.dumps(metadata_entry) + "\n")

print("📝 Metadata entry written.")

try:
    test_input = X[0].reshape(1, -1)
    test_output = model.predict(test_input)
    print(f"🧪 Sanity check passed. Example input → prediction: {test_output[0]:.2f}")
except Exception as e:
    print(f"❌ Sanity check failed: {e}")
