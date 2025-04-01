# AI Price Prediction API (with Model Registry & A/B Testing)

A production-style FastAPI service that serves a trained regression model with support for:
- Dynamic model version switching
- Health checks and metrics (Prometheus style)
- Input validation and rate limiting
- Model registry + auto validation + A/B testing
- Docker deployment + CI/CD

## Features

| Feature                | Description                                 |
|------------------------|---------------------------------------------|
| `/predict`             | Predict house price from features           |
| `/switch_model`        | Dynamically load a different model version  |
| `/status`              | Model status and switch history             |
| `/metrics`             | Prometheus-friendly metrics                 |
| Model registry         | `ml_pipeline/model_store/model_vX.pkl`               |
| Auto test              | Validate models after loading               |
| Version freeze         | Forbid usage of certain versions            |
| A/B testing            | Hash-based routing and hit counter          |
| Logging                | Trace logs and model switch logs            |

## Run Locally

```bash
uvicorn app.main:app --reload
