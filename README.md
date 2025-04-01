# Linear Regression Model Serving API

This project demonstrates a production-grade FastAPI service for serving multiple versions of a trained linear regression model. It includes features like model version switching, A/B testing, Prometheus-style metrics, request rate-limiting, automated CI/CD, and model registry with training metadata.

## Features

- FastAPI-based RESTful API
- Model versioning and dynamic hot swapping
- Prometheus-compatible metrics endpoint
- A/B testing with deterministic hashing
- Predict endpoint with input validation and rate-limiting
- Status endpoint showing current version, hit counts, errors, latency, and switch history
- Version freezing to disable specific models temporarily
- Automatic model validation during switching
- Model registry with training metadata (weights, MSE, sample size, etc.)
- ML pipeline for training, evaluation, and auto-registration
- GitHub Actions CI pipeline to trigger training and testing
- Dockerized deployment (Render / DockerHub compatible)

## Endpoints

### `POST /predict`
- Accepts input features and returns predicted value and model version.
- Validates input dimension and request frequency.

### `POST /switch_model?version=v2`
- Switches the current serving model to the specified version.
- Automatically validates model before switching.

### `GET /status`
- Returns current model version, hit counters, error count, average latency, last switch log.

### `GET /metrics`
- Exposes Prometheus-style metrics.

### `POST /freeze?version=v3`
- Freezes the given version (excluded from routing and prediction).

### `POST /unfreeze?version=v3`
- Unfreezes a previously frozen version.

## Model Registry

- Models are stored under `ml_pipeline/model_store/`.
- Metadata is recorded in `metadata.jsonl`, including training info: version, date, weights, MSE, noise level, samples.

## Training Pipeline

- Script: `ml_pipeline/train_and_register.py`
- Configurable via command-line flags:
python ml_pipeline/train_and_register.py –version v3 –samples 200 –noise 0.2

- On success, stores model + metadata + triggers validation.

## CI/CD

- GitHub Actions: `.github/workflows/ci.yaml`
- On push, runs unit tests, trains new models, and verifies pipeline.
- Render deployment uses Dockerfile.

## Project Structure
linear-regression-service/
├── app/
│   ├── api/
│   │   ├── predict.py
│   │   ├── switch_model.py
│   │   ├── freeze.py
│   │   ├── status.py
│   │   ├── metrics.py
│   │   └── healthz.py
│   ├── core/
│   │   ├── model_holder.py
│   │   ├── ab_router.py
│   │   ├── limiter.py
│   │   └── logging_utils.py
│   └── main.py
├── utils/
│   └── model_utils.py
├── ml_pipeline/
│   ├── data_generator.py
│   └── train_and_register.py
├── tests/
│   ├── test_predict.py
│   └── conftest.py
├── .github/
│   └── workflows/
│       └── ci.yaml
├── Dockerfile
├── requirements.txt
├── .dockerignore
├── .gitignore
├── pytest.ini
└── README.md

## Requirements

Install dependencies:
pip install -r requirements.txt

## Docker

Build and run locally:
docker build -t ml-api .
docker run -p 8000:8000 ml-api

## Testing

Unit tests via pytest:
pytest

CI pipeline runs all tests automatically on GitHub.

---
