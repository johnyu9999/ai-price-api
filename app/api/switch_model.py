import time, logging
from datetime import datetime
from fastapi import APIRouter, Query, HTTPException
from fastapi.responses import JSONResponse
from app.core.model_holder import model_holder
from app.core.logging_utils import log_switch_to_file
from utils.model_utils import load_model

router = APIRouter()

@router.post("/switch_model")
def switch_model(version: str = Query(..., description="Target model version")):
    if version in model_holder.frozen_versions:
        raise HTTPException(status_code=403, detail=f"Version '{version}' is frozen and cannot be used.")

    try:
        logging.info(f"🔄 Switching to model version: {version}")
        start = time.time()
        new_model = load_model(version)
        duration_ms = int((time.time() - start) * 1000)

        from_version = model_holder.model_version
        with model_holder.model_lock:
            model_holder.model = new_model
            model_holder.model_version = version
            model_holder.last_switch_info = {
                "from": from_version,
                "to": version,
                "duration_ms": duration_ms,
                "timestamp": datetime.utcnow().isoformat()
            }

        log_switch_to_file(from_version, version, duration_ms)
        return {
            "message": f"Model switched to version {version}",
            "current_version": model_holder.model_version
        }

    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Model version '{version}' not found.")
    except Exception as e:
        logging.exception("Unexpected error during model switch")
        return JSONResponse(status_code=500, content={"error": str(e)})
