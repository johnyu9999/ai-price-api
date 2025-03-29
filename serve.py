import joblib
import numpy as np
import time
import os
import uuid
import logging
import model_holder
import json

from datetime import datetime
from fastapi import FastAPI, HTTPException, Request, APIRouter, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List
from sklearn.linear_model import LinearRegression
from functools import lru_cache
from collections import defaultdict
from model_utils import load_model

MODEL_VERSION = os.getenv("MODEL_VERSION", "v1")
model_holder.model = load_model(MODEL_VERSION)
model_holder.model_version = MODEL_VERSION
rate_limit = defaultdict(list)
MAX_REQUESTS = 5
WINDOW_SECONDS = 60
app = FastAPI()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)

class PredictRequest(BaseModel):
    features: List[float]

@app.get("/")
def read_root():
    return {"status": "ok", "message": "API is live."}


@app.get("/healthz")
def health_check():
    return {"status": "healthy"}


@lru_cache(maxsize=128)
def cached_predict(feature_tuple):
    X = np.array([feature_tuple])
    return model_holder.model.predict(X)[0]

@app.post("/predict")
async def predict(request: Request, body: PredictRequest):
    ip = request.client.host
    now = time.time()

    rate_limit[ip] = [t for t in rate_limit[ip] if now - t < WINDOW_SECONDS]

    if len(rate_limit[ip]) >= MAX_REQUESTS:
        raise HTTPException(
            status_code=429, detail="Too many requests - rate limit exceeded."
        )

    rate_limit[ip].append(now)

    start_time = time.time()
    trace_id = str(uuid.uuid4())[:8]

    if len(body.features) != 3:
        raise HTTPException(
            status_code=400, detail="there must be 3 numbers in the request"
        )

    try:
        feature_tuple = tuple(body.features)
        prediction = cached_predict(feature_tuple)
        duration = (time.time() - start_time) * 1000
        logging.info(
            f"[trace:{trace_id}] {request.client.host} called /predict with input={body.features} → output={prediction:.2f} | version={model_holder.model_version} ({duration:.1f} ms)"
        )
        return {
            "predicted_price": round(prediction, 2),
            "model_version": model_holder.model_version,
            "trace_id": trace_id,
        }
    except Exception as e:
        logging.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/switch_model")
def switch_model(version: str = Query(..., description="Target model version to switch to")):
    try:
        logging.info(f"🔄 Received request to switch model to version: {version}")

        start = time.time()
        new_model = load_model(version)
        duration_ms = int((time.time() - start) * 1000)
        log_switch_to_file(model_holder.model_version, version, duration_ms)
        model_holder.model = new_model
        model_holder.model_version = version

        logging.info(f"✅ Model successfully switched to version: {version}")

        return {
            "message": f"Model switched to version {version}",
            "current_version": model_holder.model_version
        }

    except FileNotFoundError:
        logging.error(f"❌ Failed to switch model - version not found: {version}")
        raise HTTPException(status_code=404, detail=f"Model version '{version}' not found.")

    except Exception as e:
        logging.exception("❌ Unexpected error during model switch")
        return JSONResponse(status_code=500, content={"error": str(e)})

def log_switch_to_file(from_version: str, to_version: str, duration_ms: int):
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "action": "switch",
        "from": from_version,
        "to": to_version,
        "duration_ms": duration_ms
    }
    with open("model_switch_log.jsonl", "a") as f:
        f.write(json.dumps(log_entry) + "\n")

print("=== ROUTES LOADED ===")
for route in app.routes:
    print(f"{route.path} → {route.name}")

print("✅ Running from serve.py")
