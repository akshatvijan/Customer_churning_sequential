from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from app.schemas.retention import RetentionRecommendation


class ChurnPredictionResponse(BaseModel):
    """Unified churn prediction output with strategic retention recommendation."""
    customer_id: str = Field(..., description="Customer ID")
    model_name: str = Field(..., description="Model used for scoring (ANN, RNN, LSTM, Combined)")
    churn_probability: float = Field(..., ge=0.0, le=1.0, description="Predicted probability of churning")
    will_churn: bool = Field(..., description="Binary classification (probability >= 0.5)")
    risk_level: str = Field(..., description="Churn risk tier (Low, Medium, High)")
    customer_value: str = Field(..., description="Customer value tier (Low, Medium, High)")
    retention: RetentionRecommendation = Field(..., description="Actionable retention recommendation")
    top_risk_factors: Optional[List[str]] = Field(default_factory=list, description="Key behavioural/contract drivers")
    inference_time_ms: Optional[float] = Field(None, description="Inference latency in milliseconds")


class SequencePredictionResponse(BaseModel):
    """Sequential RNN/LSTM churn prediction from customer behavioural trajectory."""
    customer_id: str
    model_name: str
    churn_probability: float
    will_churn: bool
    risk_level: str
    trajectory_trend: str = Field(..., description="Trend direction: 'Deteriorating', 'Stable', or 'Improving'")
    retention: RetentionRecommendation
    sequence_length: int
    inference_time_ms: Optional[float]


class CombinedPredictionResponse(BaseModel):
    """Advanced combined architecture: ANN (Current Profile) + LSTM (Behavioural History)."""
    customer_id: str
    ann_probability: float = Field(..., description="Static profile churn probability")
    lstm_probability: float = Field(..., description="Behavioural sequence churn probability")
    combined_churn_probability: float = Field(..., description="Ensemble combined churn probability")
    will_churn: bool
    risk_level: str
    customer_value: str
    retention: RetentionRecommendation
    summary: str
    inference_time_ms: Optional[float]


class BatchPredictionSummary(BaseModel):
    """Aggregate overview of a batch churn scoring run."""
    total_customers: int
    high_risk_count: int
    medium_risk_count: int
    low_risk_count: int
    projected_churn_rate: float
    total_retention_budget_needed: float
    results: List[ChurnPredictionResponse]
