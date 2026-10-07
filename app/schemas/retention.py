from typing import Optional
from pydantic import BaseModel, Field


class RetentionRecommendation(BaseModel):
    """Actionable retention recommendation based on churn risk and customer value."""
    risk_level: str = Field(..., description="Churn risk tier: Low, Medium, High")
    customer_value: str = Field(..., description="Customer value tier: Low, Medium, High")
    recommended_action: str = Field(..., description="Action code: e.g., PRIORITY_RETENTION, PERSONALIZED_OFFER")
    recommended_reward: str = Field(..., description="Concrete customer incentive, e.g. '20% Discount'")
    reward_cost: float = Field(..., description="Direct incentive budget cost in USD")
    description: Optional[str] = Field(None, description="Strategic rationale for retention team")


class RetentionStrategyRequest(BaseModel):
    """Direct request to evaluate retention strategy given probability and historical value."""
    churn_probability: float = Field(..., ge=0.0, le=1.0, description="Predicted churn probability")
    totalcharges: float = Field(..., ge=0.0, description="Total historical customer spend")


class ROIRequest(BaseModel):
    """Parameters to evaluate expected ROI and net business impact for a retention campaign."""
    customers_targeted: int = Field(100, ge=1, description="Number of high/medium risk customers targeted")
    expected_incremental_retention: float = Field(0.20, ge=0.0, le=1.0, description="Estimated incremental retention rate (0.0 to 1.0)")
    average_customer_value: float = Field(2500.0, ge=0.0, description="Average customer lifetime value in USD")
    total_reward_cost: float = Field(2500.0, ge=0.0, description="Sum of reward costs allocated")
    campaign_cost: float = Field(500.0, ge=0.0, description="Operational & communication overhead")


class ROIResponse(BaseModel):
    """Campaign business impact projection and calculated return on investment."""
    customers_targeted: int
    expected_incremental_retention: float
    average_customer_value: float
    potential_retained_value: float
    total_reward_cost: float
    campaign_cost: float
    total_campaign_cost: float
    net_business_impact: float
    roi_percentage: float
