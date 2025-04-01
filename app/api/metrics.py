from fastapi import APIRouter
from fastapi.responses import PlainTextResponse
from app.core.model_holder import model_holder

router = APIRouter()

@router.get("/metrics", response_class=PlainTextResponse)
def metrics():
    metrics_text = f"""
# HELP predict_error_total Total prediction errors
# TYPE predict_error_total counter
predict_error_total {model_holder.error_count}

# HELP avg_predict_latency_ms Average prediction latency
# TYPE avg_predict_latency_ms gauge
avg_predict_latency_ms {round(model_holder.total_latency / model_holder.total_requests, 2) if model_holder.total_requests else 0.0}
"""
    return metrics_text.strip()
