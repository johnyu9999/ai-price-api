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

MODEL_VERSION = "v1.0-fallback"

# 初始化日志系统
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# 检查是否存在模型文件，如果没有则训练一个模型
model_path = "model.pkl"

if not os.path.exists(model_path):
    logging.info("🚧 model.pkl not found. Training a simple fallback model...")
    np.random.seed(42)
    X = np.random.rand(100, 3)
    weights = np.array([1.5, -2.0, 3.0])
    y = X @ weights + 4.2 + np.random.randn(100) * 0.1
    model = LinearRegression()
    model.fit(X, y)
    joblib.dump(model, model_path)
    logging.info("✅ Fallback model trained and saved to model.pkl")
else:
    model = joblib.load(model_path)
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
    start_time = time.time()
    trace_id = str(uuid.uuid4())[:8]  # 简洁一点

    if len(body.features) != 3:
        raise HTTPException(status_code=400, detail="❌ 输入 features 必须包含 3 个数字。")

    try:
        feature_tuple = tuple(body.features)
        prediction = cached_predict(feature_tuple)
        duration = (time.time() - start_time) * 1000
        logging.info(f"[trace:{trace_id}] {request.client.host} called /predict with input={body.features} → output={y_pred[0]:.2f} | version={MODEL_VERSION} ({duration:.1f} ms)")
        return {
                "predicted_price": round(prediction, 2),
                "model_version": MODEL_VERSION,
                "trace_id": trace_id
        }
    except Exception as e:
        logging.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

