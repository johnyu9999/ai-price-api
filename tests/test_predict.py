import sys
import os
import pytest
from fastapi.testclient import TestClient

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from main import app

from app.core.model_holder import model_holder
from utils.model_utils import load_model

client = TestClient(app)

def test_predict_valid_input(test_client):
    response = test_client.post("/predict", json={"features": [0.1, 0.2, 0.3]})
    print("response: ", response.json())
    assert response.status_code == 200
    data = response.json()
    assert "predicted_price" in data
    assert isinstance(data["predicted_price"], float)

def test_predict_missing_features():
    response = client.post("/predict", json={})
    assert response.status_code == 422

def test_predict_wrong_feature_count():
    response = client.post("/predict", json={"features": [0.1]})
    assert response.status_code == 400
    assert "Expected 3 features" in response.text

def test_predict_rate_limit():
    for _ in range(5):
        client.post("/predict", json={"features": [0.1, 0.2, 0.3]})
    response = client.post("/predict", json={"features": [0.1, 0.2, 0.3]})
    assert response.status_code == 429
