# app/api/freeze.py

from fastapi import APIRouter, Query, HTTPException
from app.core.model_holder import model_holder


router = APIRouter()

@router.post("/freeze_version")
def freeze_model_version(version: str = Query(..., description="model version to freeze")):
    if version not in model_holder.model_versions:
        raise HTTPException(status_code=404, detail="model is not found")
    model_holder.frozen_versions.add(version)
    return {"message": f"Model with version {version} is frozen."}

@router.post("/unfreeze_version")
def unfreeze_model_version(version: str = Query(..., description="model version to unfreeze")):
    if version not in model_holder.frozen_versions:
        return {"message": f"Model with version {version} is not frozen."}
    model_holder.frozen_versions.remove(version)
    return {"message": f"Model with version {version} is unfrozen."}
