from typing import Optional, List
from pydantic import BaseModel, Field


class CustomerInput(BaseModel):
    """Demographic, service, billing, and behavioral attributes for a customer."""
    customer_id: Optional[str] = Field("CUST10291", description="Unique identifier for customer")
    signup_date: Optional[str] = Field("2023-01-15", description="Customer signup date (YYYY-MM-DD)")
    age: int = Field(38, ge=18, le=100, description="Customer age")
    gender: str = Field("Female", description="Gender (Female / Male)")
    annual_income: float = Field(65000.0, ge=0.0, description="Annual income in USD")
    education: str = Field("college", description="Education level (high_school, college, bachelor, master, phd)")
    marital_status: str = Field("married", description="Marital status (single, married, divorced, widowed)")
    dependents: int = Field(1, ge=0, description="Number of dependents")
    tenure: int = Field(12, ge=0, description="Tenure in months")
    contract: str = Field("month-to-month", description="Contract term (month-to-month, one_year, two_year)")
    payment_method: str = Field("electronic_check", description="Payment method (electronic_check, credit_card, bank_transfer, mailed_check)")
    paperless_billing: str = Field("Yes", description="Paperless billing (Yes / No)")
    senior_citizen: int = Field(0, ge=0, le=1, description="Senior citizen indicator (0 or 1)")
    monthlycharges: float = Field(85.5, ge=0.0, description="Monthly charges in USD")
    totalcharges: float = Field(1026.0, ge=0.0, description="Total historical charges in USD")
    num_services: int = Field(3, ge=1, le=10, description="Total number of subscribed services")
    has_phone_service: int = Field(1, ge=0, le=1, description="Phone service (0 or 1)")
    has_internet_service: int = Field(1, ge=0, le=1, description="Internet service (0 or 1)")
    has_online_security: int = Field(0, ge=0, le=1, description="Online security add-on (0 or 1)")
    has_online_backup: int = Field(1, ge=0, le=1, description="Online backup add-on (0 or 1)")
    has_device_protection: int = Field(0, ge=0, le=1, description="Device protection add-on (0 or 1)")
    has_tech_support: int = Field(0, ge=0, le=1, description="Tech support add-on (0 or 1)")
    has_streaming_tv: int = Field(1, ge=0, le=1, description="Streaming TV (0 or 1)")
    has_streaming_movies: int = Field(1, ge=0, le=1, description="Streaming Movies (0 or 1)")
    customer_satisfaction: float = Field(6.0, ge=1.0, le=10.0, description="Satisfaction score (1-10)")
    num_complaints: float = Field(1.0, ge=0.0, description="Number of recorded complaints")
    num_service_calls: float = Field(2.0, ge=0.0, description="Number of customer service calls")
    late_payments: float = Field(1.0, ge=0.0, description="Number of late payments")
    avg_monthly_gb: float = Field(55.0, ge=0.0, description="Average monthly data usage in GB")
    days_since_last_interaction: float = Field(12.0, ge=0.0, description="Days since last customer interaction")
    credit_score: Optional[float] = Field(640.0, ge=300.0, le=850.0, description="Customer credit score")


class BatchCustomerInput(BaseModel):
    """Batch list of customer inputs for bulk evaluation."""
    customers: List[CustomerInput]
