# app/main.py
from fastapi import FastAPI
from contextlib import asynccontextmanager

from app.api import (
    predict,
    status,
    switch_model,
    freeze,
    metrics,
)

from app.core.model_holder import model_holder
from utils.model_utils import load_model


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        model = load_model("v1")
        model_holder.model = model
        model_holder.model_version = "v1"
        model_holder.input_dims = model.coef_.shape[0]
        print("Model loaded in lifespan")
    except Exception as e:
        print("Failed to load model during startup:", e)
        model_holder.model = None

    yield  # App is running


app = FastAPI(lifespan=lifespan)

app.include_router(predict.router)
app.include_router(status.router)
app.include_router(switch_model.router)
app.include_router(freeze.router)
app.include_router(metrics.router)
