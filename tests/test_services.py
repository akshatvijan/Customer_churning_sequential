import pytest
import numpy as np

from app.services.retention_engine import (
    assign_churn_risk,
    assign_customer_value,
    recommend_retention_action,
    calculate_roi
)
from app.services.preprocessor import ChurnPreprocessor
from app.schemas.sequence import MonthlyRecord


def test_assign_churn_risk():
    assert assign_churn_risk(0.20) == "Low"
    assert assign_churn_risk(0.55) == "Medium"
    assert assign_churn_risk(0.85) == "High"


def test_assign_customer_value():
    assert assign_customer_value(1000.0) == "Low"
    assert assign_customer_value(2500.0) == "Medium"
    assert assign_customer_value(5000.0) == "High"


def test_recommend_retention_action():
    # Low churn probability gets NO_INTERVENTION regardless of spend
    rec_low = recommend_retention_action(churn_probability=0.25, totalcharges=4500.0)
    assert rec_low.recommended_action == "NO_INTERVENTION"
    assert rec_low.reward_cost == 0.0

    # High risk & High value gets PRIORITY_RETENTION
    rec_high = recommend_retention_action(churn_probability=0.85, totalcharges=4500.0)
    assert rec_high.recommended_action == "PRIORITY_RETENTION"
    assert rec_high.reward_cost == 50.0

    # High risk & Low value gets LOW_COST_RETENTION
    rec_low_val = recommend_retention_action(churn_probability=0.85, totalcharges=800.0)
    assert rec_low_val.recommended_action == "LOW_COST_RETENTION"
    assert rec_low_val.reward_cost == 10.0


def test_calculate_roi():
    roi = calculate_roi(
        customers_targeted=200,
        expected_incremental_retention=0.25,
        average_customer_value=2000.0,
        total_reward_cost=5000.0,
        campaign_cost=1000.0
    )
    # potential_retained = 200 * 0.25 * 2000 = 100,000
    # total_cost = 5000 + 1000 = 6,000
    # net_impact = 100,000 - 6,000 = 94,000
    # roi = (94000 / 6000) * 100 = 1566.67%
    assert roi.potential_retained_value == 100000.0
    assert roi.total_campaign_cost == 6000.0
    assert roi.net_business_impact == 94000.0
    assert roi.roi_percentage == pytest.approx(1566.67, 0.01)


def test_preprocessor_features():
    d = {
        "tenure": 12,
        "totalcharges": 1200.0,
        "monthlycharges": 100.0,
        "annual_income": 60000.0,
        "num_services": 3,
        "num_complaints": 2,
        "contract": "month-to-month"
    }
    vec, risks = ChurnPreprocessor.extract_features_from_dict(d)
    assert isinstance(vec, np.ndarray)
    assert len(vec) > 30
    assert any("Month-to-Month" in r for r in risks)
    assert any("Complaint" in r for r in risks)


def test_preprocessor_sequence():
    records = [
        MonthlyRecord(month=1, avg_monthly_gb=80, monthlycharges=70, num_complaints=0, num_service_calls=1, late_payments=0, days_since_last_interaction=10),
        MonthlyRecord(month=2, avg_monthly_gb=50, monthlycharges=70, num_complaints=2, num_service_calls=2, late_payments=1, days_since_last_interaction=20),
    ]
    mat, trend = ChurnPreprocessor.preprocess_sequence(records)
    assert mat.shape == (1, 2, 6)
    assert "Deteriorating" in trend
