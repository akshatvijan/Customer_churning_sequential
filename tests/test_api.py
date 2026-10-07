import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health():
    res = client.get("/api/v1/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert "pytorch_version" in data
    assert "models" in data
    assert "rnn" in data["models"]
    assert "lstm" in data["models"]
    assert "ann" in data["models"]
    assert "sequence_scaler" in data["models"]


def test_predict_ann():
    payload = {
        "customer_id": "CUST_TEST1",
        "age": 42,
        "gender": "Female",
        "annual_income": 60000.0,
        "education": "college",
        "marital_status": "married",
        "dependents": 1,
        "tenure": 12,
        "contract": "month-to-month",
        "payment_method": "electronic_check",
        "paperless_billing": "Yes",
        "senior_citizen": 0,
        "monthlycharges": 85.0,
        "totalcharges": 1020.0,
        "num_services": 3,
        "customer_satisfaction": 5.0,
        "num_complaints": 2.0,
        "num_service_calls": 3.0,
        "late_payments": 1.0,
        "avg_monthly_gb": 45.0,
        "days_since_last_interaction": 15.0
    }
    res = client.post("/api/v1/predict/ann", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert 0.0 <= data["churn_probability"] <= 1.0
    assert data["risk_level"] in ["Low", "Medium", "High"]
    assert "retention" in data
    assert "recommended_action" in data["retention"]


def test_predict_rnn_and_lstm():
    seq_payload = {
        "customer_id": "CUST_SEQ_TEST",
        "sequence": [
            {"month": 1, "avg_monthly_gb": 80.0, "monthlycharges": 70.0, "num_complaints": 0, "num_service_calls": 1, "late_payments": 0, "days_since_last_interaction": 10},
            {"month": 2, "avg_monthly_gb": 70.0, "monthlycharges": 70.0, "num_complaints": 1, "num_service_calls": 1, "late_payments": 0, "days_since_last_interaction": 15},
            {"month": 3, "avg_monthly_gb": 60.0, "monthlycharges": 75.0, "num_complaints": 1, "num_service_calls": 2, "late_payments": 0, "days_since_last_interaction": 20},
            {"month": 4, "avg_monthly_gb": 45.0, "monthlycharges": 80.0, "num_complaints": 2, "num_service_calls": 3, "late_payments": 1, "days_since_last_interaction": 30},
            {"month": 5, "avg_monthly_gb": 30.0, "monthlycharges": 85.0, "num_complaints": 3, "num_service_calls": 4, "late_payments": 1, "days_since_last_interaction": 40}
        ]
    }
    # RNN
    res_rnn = client.post("/api/v1/predict/rnn", json=seq_payload)
    assert res_rnn.status_code == 200
    assert "churn_probability" in res_rnn.json()

    # LSTM
    res_lstm = client.post("/api/v1/predict/lstm", json=seq_payload)
    assert res_lstm.status_code == 200
    assert "churn_probability" in res_lstm.json()


def test_retention_recommend():
    res = client.post("/api/v1/retention/recommend", json={"churn_probability": 0.85, "totalcharges": 4500.0})
    assert res.status_code == 200
    data = res.json()
    assert data["risk_level"] == "High"
    assert data["customer_value"] == "High"
    assert data["recommended_action"] == "PRIORITY_RETENTION"


def test_retention_roi():
    res = client.post("/api/v1/retention/roi", json={
        "customers_targeted": 100,
        "expected_incremental_retention": 0.20,
        "average_customer_value": 3000.0,
        "total_reward_cost": 2000.0,
        "campaign_cost": 500.0
    })
    assert res.status_code == 200
    data = res.json()
    assert data["potential_retained_value"] == 60000.0
    assert data["net_business_impact"] == 57500.0
    assert data["roi_percentage"] == 2300.0
