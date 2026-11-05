
from fastapi import APIRouter, HTTPException

from app.services.qdrant_service import check_qdrant_connection


router = APIRouter(
    prefix="/health",
    tags=["Health"],
)


@router.get("")
def health_check():
    return {
        "status": "healthy",
        "service": "enterprise-ai-copilot",
    }


@router.get("/qdrant")
def qdrant_health_check():
    try:
        return check_qdrant_connection()

    except Exception:
        raise HTTPException(
            status_code=503,
            detail="Qdrant service is unavailable",
        )