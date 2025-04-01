from fastapi import APIRouter
from app.core.model_holder import model_holder

router = APIRouter()

@router.get("/status")
def model_status():
    info = {
        "current_version": model_holder.model_version,
        "available_versions": sorted(model_holder.available_versions),
        "frozen_versions": sorted(model_holder.frozen_versions),
        "hit_counter": model_holder.hit_counter,
        "last_switch": model_holder.last_switch_info,
        "avg_latency_ms": round(model_holder.total_latency / model_holder.total_requests, 2)
        if model_holder.total_requests > 0 else None,
        "total_errors": model_holder.error_count
    }
    return info
