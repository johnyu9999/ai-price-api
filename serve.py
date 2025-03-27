from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import List
import joblib
import numpy as np
import time
import logging

# 初始化日志系统
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# 加载模型
model = joblib.load("model.pkl")

# 定义输入格式
class PredictRequest(BaseModel):
    features: List[float]

# 初始化 FastAPI 应用
app = FastAPI()

@app.post("/predict")
async def predict(request: Request, body: PredictRequest):
    start_time = time.time()

    # 校验输入长度
    if len(body.features) != 3:
        raise HTTPException(status_code=400, detail="❌ 输入 features 必须包含 3 个数字。")

    try:
        X = np.array([body.features])
        y_pred = model.predict(X)
        duration = (time.time() - start_time) * 1000  # ms

        # 日志输出
        logging.info(f"{request.client.host} called /predict with input={body.features} → output={y_pred[0]:.2f} ({duration:.1f} ms)")

        return {"predicted_price": round(y_pred[0], 2)}

    except Exception as e:
        logging.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
