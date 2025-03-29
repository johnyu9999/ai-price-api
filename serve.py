import joblib
import numpy as np
import time
import os
import uuid
import logging
import model_holder
import json

from fastapi.responses import PlainTextResponse    
from datetime import datetime
from fastapi import FastAPI, HTTPException, Request, APIRouter, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List
from sklearn.linear_model import LinearRegression
from functools import lru_cache
from collections import defaultdict
from model_utils import load_model

model_holder.model_versions["v1"] = load_model("v1")
model_holder.model = model_holder.model_versions["v1"]
model_holder.model_versions["v2"] = load_model("v2")
model_holder.model_versions["v3"] = load_model("v3")
model_holder.model_versions["v4"] = load_model("v4")
model_holder.model_hit_counter.setdefault("v1", 0)
model_holder.model_hit_counter.setdefault("v2", 0)
model_holder.model_hit_counter.setdefault("v3", 0)
model_holder.model_hit_counter.setdefault("v4", 0)

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
    X = np.array(feature_tuple, dtype=np.float32).reshape(1, -1)
    with model_holder.model_lock:
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


    try:
        feature_tuple = tuple(body.features)
        prediction = cached_predict(feature_tuple)
        duration = (time.time() - start_time) * 1000
        with model_holder.model_lock:
            model_holder.prediction_durations_ms.append(duration)
            model_holder.model_hit_counter[model_holder.current_version] += 1
        pred_value = float(prediction[0]) if isinstance(prediction, (np.ndarray, list)) else float(prediction)
        logging.info(
            f"[trace:{trace_id}] {request.client.host} called /predict with input={body.features} → output={pred_value:.2f} | version={model_holder.current_version} ({duration:.1f} ms)"
        )
        model_holder.predict_latency_total[model_holder.current_version] += duration
        model_holder.predict_latency_count[model_holder.current_version] += 1
        return {
            "predicted_price": round(pred_value, 2),
            "current_version": model_holder.current_version,
            "trace_id": trace_id,
        }
    except Exception as e:
        model_holder.predict_error_total += 1
        logging.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/switch_model")
def switch_model(version: str = Query(..., description="Target model version to switch to")):
    try:
        logging.info(f"🔄 Received request to switch model to version: {version}")

        start = time.time()
        with model_holder.model_lock:
            old_version = model_holder.current_version
            duration_ms = int((time.time() - start) * 1000) 
            if version in model_holder.frozen_versions:
                raise HTTPException(status_code=403, detail=f"Version '{version}' is frozen.")
            new_model = load_model(version)
            model_holder.model = new_model
            model_holder.current_version = version
            model_holder.last_switch_info = {
                "from": old_version,
                "to": version,
                "duration_ms": duration_ms,
                "timestamp": datetime.utcnow().isoformat()
            }
        duration_ms = int((time.time() - start) * 1000)
        log_switch_to_file(model_holder.current_version, version, duration_ms)

        logging.info(f"✅ Model successfully switched to version: {version}")
        try:
            test_input = get_test_input(version)
            new_model.predict(test_input)
            logging.info(f"✅ Post-load validation passed for version: {version}")
        except Exception as e:
            logging.error(f"❌ Model validation failed after loading version: {version} — {e}")
            raise HTTPException(status_code=500, detail=f"Model validation failed: {str(e)}")

        return {
            "message": f"Model switched to version {version}",
            "current_version": model_holder.current_version
        }

    except FileNotFoundError:
        logging.error(f"❌ Failed to switch model - version not found: {version}")
        raise HTTPException(status_code=404, detail=f"Model version '{version}' not found.")

    except Exception as e:
        logging.exception("❌ Unexpected error during model switch")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/debug_model_info")
def debug_model_info():
    return {
        "version": model_holder.current_version,
        "coef": model_holder.model.coef_.tolist(),
        "bias": model_holder.model.intercept_.tolist(),
        "coef_shape": np.array(model_holder.model.coef_).shape
    }

@app.get("/status")
def status():
    uptime = datetime.utcnow() - model_holder.startup_time
    avg_latency = (
        sum(model_holder.prediction_durations_ms) / len(model_holder.prediction_durations_ms)
        if model_holder.prediction_durations_ms else None
    )
    return {
        "status": "ok",
        "uptime": str(uptime).split(".")[0],
        "avg_prediction_latency_ms": round(avg_latency, 2) if avg_latency else "N/A",
        "last_switch": model_holder.last_switch_info,
        "loaded_versions": list(model_holder.model_versions.keys()),
        "model_hit_counter": model_holder.model_hit_counter
    }

@app.post("/predict_ab")
async def predict_ab(
    request: Request,
    body: PredictRequest,
    user_id: str = Query(...),
    version: str = Query(None, description="Optional. Model version. If omitted, A/B split applies.")
):
    trace_id = str(uuid.uuid4())[:8]
    
    try:
        with model_holder.model_lock:
            available_versions = [
                v for v in model_holder.model_versions.keys()
                if v not in model_holder.frozen_versions
            ]
            chosen_version = hash_user_to_version(user_id, available_versions)
            model = model_holder.model_versions[chosen_version]
            if not model:
                raise HTTPException(status_code=404, detail=f"Model version '{chosen_version}' not found.")

            model_holder.model_hit_counter[chosen_version] += 1
        X = np.array([tuple(body.features)])
        prediction = model.predict(X)[0]

        logging.info(
            f"[trace:{trace_id}] /predict_ab auto-routed to {chosen_version} → {float(prediction):.2f}"
        )

        return {
            "predicted_price": round(float(prediction), 2),
            "current_version": chosen_version,
            "trace_id": trace_id
        }

    except Exception as e:
        logging.exception("Prediction failed in /predict_ab")
        raise HTTPException(status_code=500, detail=f"AB Prediction error: {str(e)}")

@app.get("/metrics", response_class=PlainTextResponse)
def metrics():
    lines = []

    with model_holder.model_lock:
        for version, count in model_holder.model_hit_counter.items():
            lines.append(f'model_hit_counter{{version="{version}"}} {count}')
        
        lines.append(f"active_models {len(model_holder.model_versions)}")
        lines.append(f"predict_error_total {model_holder.predict_error_total}")
        for version in model_holder.predict_latency_total:
            total = model_holder.predict_latency_total[version]
            count = model_holder.predict_latency_count[version]
            avg_latency = total / count if count else 0
            lines.append(f'avg_predict_latency_ms{{version="{version}"}} {avg_latency:.2f}')

    return "\n".join(lines)

@app.post("/freeze_version")
def freeze_version(version: str = Query(...)):
    with model_holder.model_lock:
        model_holder.frozen_versions.add(version)
    return {"message": f"Version {version} is now frozen."}

import hashlib

def hash_user_to_version(user_id: str, versions: List[str]) -> str:
    hashed = int(hashlib.sha256(user_id.encode()).hexdigest(), 16)
    return versions[hashed % len(versions)]

def get_test_input(version: str) -> np.ndarray:
    if version in ("v1", "v4"):
        return np.array([[0.1, 0.2, 0.3]])
    elif version in ("v2", "v3"):
        return np.array([[0.1]])
    else:
        raise ValueError(f"No test input defined for version: {version}")

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

import random

def pick_model_version_by_ratio():
    r = random.random()  # 0.0 ~ 1.0
    return "v1" if r < 0.8 else "v2"

print("=== ROUTES LOADED ===")
for route in app.routes:
    print(f"{route.path} → {route.name}")

print("✅ Running from serve.py")
