import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, JSONResponse

from app.api import api_router

app = FastAPI(
    title="Customer Churn Prediction & Retention Strategy API",
    description="""
## Enterprise Customer Churn & Sequential Behavior Analytics Platform

This API delivers production-grade customer churn intelligence:
- **PyTorch ANN Model**: Tabular inference evaluating demographics, contracts, billing, and behavioral attributes.
- **PyTorch RNN & LSTM Sequential Models**: Evaluates 5-month temporal customer trajectories to identify deteriorating engagement.
- **Combined Architecture (Section 14 & 20)**: Static profile + sequential history ensemble.
- **Retention Strategy & ROI Engine**: Dynamic risk categorization, customer value segmentation, configurable reward recommendations, and financial campaign ROI forecasting.

Interactive Swagger documentation is available at `/docs`.
""",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for cross-origin frontend support
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include all API v1 endpoints
app.include_router(api_router)


@app.get("/", include_in_schema=False)
async def root():
    """Redirect root access to API docs or UI if available."""
    return {
        "message": "Welcome to Customer Churn Sequential Prediction & Retention Engine API",
        "documentation": "/docs",
        "health": "/api/v1/health",
        "endpoints": {
            "predict_ann": "/api/v1/predict/ann",
            "predict_rnn": "/api/v1/predict/rnn",
            "predict_lstm": "/api/v1/predict/lstm",
            "predict_combined": "/api/v1/predict/combined",
            "predict_batch": "/api/v1/predict/batch",
            "upload_csv": "/api/v1/predict/upload-csv",
            "retention_recommend": "/api/v1/retention/recommend",
            "retention_roi": "/api/v1/retention/roi"
        }
    }


def create_app() -> FastAPI:
    return app


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
