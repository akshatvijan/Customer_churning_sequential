from fastapi import APIRouter
from app.api.health import router as health_router
from app.api.predict import router as predict_router
from app.api.retention import router as retention_router

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(health_router)
api_router.include_router(predict_router)
api_router.include_router(retention_router)

__all__ = ["api_router"]
