from fastapi import APIRouter
from typing import Dict, Any
import torch
import sys

router = APIRouter(tags=["Health"])


@router.get("/health", summary="Health & System Diagnostics")
async def health_check() -> Dict[str, Any]:
    """Check API operational status, PyTorch device availability, and Python version."""
    return {
        "status": "healthy",
        "service": "Customer Churn Prediction & Retention API",
        "version": "1.0.0",
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "python_version": sys.version.split(" ")[0]
    }
