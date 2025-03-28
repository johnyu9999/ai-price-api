from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import List
import joblib
import numpy as np
import time
import logging
import os
from sklearn.linear_model import LinearRegression
import uuid
from functools import lru_cache
from collections import defaultdict
import time

def load_model(version: str):
    model_path = f"model_registry/model_{version}.pkl"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found for version '{version}': {model_path}")
    
    model = joblib.load(model_path)
    return model


# 初始化日志系统
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)

# 检查是否存在模型文件，如果没有则训练一个模型

rate_limit = defaultdict(list)
MAX_REQUESTS = 5
WINDOW_SECONDS = 60
MODEL_VERSION = os.getenv("MODEL_VERSION", "v1")
model = load_model(MODEL_VERSION)
logging.info("✅ model.pkl loaded.")


# 定义输入格式
class PredictRequest(BaseModel):
    features: List[float]


# 初始化 FastAPI 应用
app = FastAPI()


@app.get("/")
def read_root():
    return {"status": "ok", "message": "API is live."}


@app.get("/healthz")
def health_check():
    return {"status": "healthy"}


@lru_cache(maxsize=128)
def cached_predict(feature_tuple):
    X = np.array([feature_tuple])
    return model.predict(X)[0]


@app.post("/predict")
async def predict(request: Request, body: PredictRequest):
    ip = request.client.host
    now = time.time()

    # 清除过期时间戳
    rate_limit[ip] = [t for t in rate_limit[ip] if now - t < WINDOW_SECONDS]

    if len(rate_limit[ip]) >= MAX_REQUESTS:
        raise HTTPException(
            status_code=429, detail="Too many requests - rate limit exceeded."
        )

    # 记录当前请求时间
    rate_limit[ip].append(now)

    start_time = time.time()
    trace_id = str(uuid.uuid4())[:8]  # 简洁一点

    if len(body.features) != 3:
        raise HTTPException(
            status_code=400, detail="❌ 输入 features 必须包含 3 个数字。"
        )

    try:
        feature_tuple = tuple(body.features)
        prediction = cached_predict(feature_tuple)
        duration = (time.time() - start_time) * 1000
        logging.info(
            f"[trace:{trace_id}] {request.client.host} called /predict with input={body.features} → output={prediction:.2f} | version={MODEL_VERSION} ({duration:.1f} ms)"
        )
        return {
            "predicted_price": round(prediction, 2),
            "model_version": MODEL_VERSION,
            "trace_id": trace_id,
        }
    except Exception as e:
        logging.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
