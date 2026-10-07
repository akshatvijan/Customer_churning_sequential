from fastapi import APIRouter
from app.schemas.retention import (
    RetentionRecommendation,
    RetentionStrategyRequest,
    ROIRequest,
    ROIResponse
)
from app.services.retention_engine import (
    recommend_retention_action,
    calculate_roi
)

router = APIRouter(prefix="/retention", tags=["Retention Strategy & ROI"])


@router.post("/recommend", response_model=RetentionRecommendation, summary="Generate Retention Action & Reward")
async def get_retention_recommendation(payload: RetentionStrategyRequest) -> RetentionRecommendation:
    """Evaluate customer churn probability and customer spend tier to recommend optimal retention reward."""
    return recommend_retention_action(
        churn_probability=payload.churn_probability,
        totalcharges=payload.totalcharges
    )


@router.post("/roi", response_model=ROIResponse, summary="Simulate Campaign ROI & Financial Impact")
async def simulate_roi(payload: ROIRequest) -> ROIResponse:
    """Simulate business impact, net financial savings, and return on investment (ROI) for a retention campaign."""
    return calculate_roi(
        customers_targeted=payload.customers_targeted,
        expected_incremental_retention=payload.expected_incremental_retention,
        average_customer_value=payload.average_customer_value,
        total_reward_cost=payload.total_reward_cost,
        campaign_cost=payload.campaign_cost
    )
