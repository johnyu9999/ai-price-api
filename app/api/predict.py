import time, uuid
from typing import List
from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel
import numpy as np
from app.core.model_holder import model_holder
from app.core.limiter import rate_limiter
from app.core.logging_utils import log_predict_latency

router = APIRouter()

class PredictRequest(BaseModel):
    features: List[float]

import traceback

def cached_predict(feature_tuple):
    try:
        X = np.array([feature_tuple])
        return model_holder.model.predict(X)[0]

    except Exception as e:
        print("Exception occurred in cached_predict():", e)
        traceback.print_exc()
        raise

@router.post("/predict")
async def predict(request: Request, body: PredictRequest):
    ip = request.client.host
    now = time.time()

    if not rate_limiter.allow(ip, now):
        raise HTTPException(status_code=429, detail="Too many requests")

    rate_limiter.record(ip, now)
    start_time = time.time()

    if len(body.features) != model_holder.input_dims:
        raise HTTPException(status_code=400, detail=f"Expected {model_holder.input_dims} features")

    try:
        feature_tuple = tuple(body.features)
        prediction = cached_predict(feature_tuple)

        log_predict_latency(time.time() - start_time * 1000)
        model_holder.hit_counter[model_holder.model_version] += 1

        return {
            "predicted_price": round(float(prediction), 2),
            "model_version": model_holder.model_version,
            "trace_id": str(uuid.uuid4())[:8],
        }

    except Exception as e:
        model_holder.error_count += 1
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
