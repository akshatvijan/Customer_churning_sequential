from typing import List, Optional
from pydantic import BaseModel, Field


class MonthlyRecord(BaseModel):
    """Behavioral metrics observed for a customer in a single month."""
    month: int = Field(1, ge=1, le=12, description="Month index (1 to 5)")
    avg_monthly_gb: float = Field(..., ge=0.0, description="Data usage in GB")
    monthlycharges: float = Field(..., ge=0.0, description="Monthly charges billed")
    num_complaints: float = Field(0.0, ge=0.0, description="Complaints filed that month")
    num_service_calls: float = Field(0.0, ge=0.0, description="Service calls made that month")
    late_payments: float = Field(0.0, ge=0.0, description="Late payments recorded that month")
    days_since_last_interaction: float = Field(15.0, ge=0.0, description="Days since last touchpoint")


class CustomerSequenceInput(BaseModel):
    """Historical 5-month behavioural sequence for sequential RNN/LSTM evaluation."""
    customer_id: Optional[str] = Field("CUST10291", description="Customer identifier")
    sequence: List[MonthlyRecord] = Field(
        ...,
        min_length=2,
        max_length=12,
        description="Temporal monthly sequence. Standard window is 5 months."
    )


class CombinedInput(BaseModel):
    """Combined profile containing current static attributes and temporal behaviour sequence."""
    customer: "CustomerInput"
    history: Optional[CustomerSequenceInput] = None


from app.schemas.customer import CustomerInput
CombinedInput.model_rebuild()
