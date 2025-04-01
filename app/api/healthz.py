from fastapi import APIRouter

router = APIRouter()

@router.get("/")
def read_root():
    return {"status": "ok", "message": "API is live."}

@router.get("/healthz")
def health_check():
    return {"status": "healthy"}
