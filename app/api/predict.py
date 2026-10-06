import io
import pandas as pd
from typing import List
from fastapi import APIRouter, UploadFile, File, HTTPException

from app.schemas.customer import CustomerInput, BatchCustomerInput
from app.schemas.sequence import CustomerSequenceInput, CombinedInput
from app.schemas.prediction import (
    ChurnPredictionResponse,
    SequencePredictionResponse,
    CombinedPredictionResponse,
    BatchPredictionSummary
)
from app.services.predictor import ChurnPredictorService

router = APIRouter(prefix="/predict", tags=["Prediction"])


@router.post("/ann", response_model=ChurnPredictionResponse, summary="Predict Churn with PyTorch ANN")
async def predict_ann(customer: CustomerInput) -> ChurnPredictionResponse:
    """Predict churn probability using the Deep ANN on customer demographic, service, and billing profile."""
    return ChurnPredictorService.predict_ann(customer)


@router.post("/rnn", response_model=SequencePredictionResponse, summary="Predict Churn with Vanilla RNN")
async def predict_rnn(sequence_data: CustomerSequenceInput) -> SequencePredictionResponse:
    """Predict churn from a customer's temporal sequence of behavioural metrics using PyTorch RNN."""
    return ChurnPredictorService.predict_sequence(sequence_data, model_type="RNN")


@router.post("/lstm", response_model=SequencePredictionResponse, summary="Predict Churn with LSTM")
async def predict_lstm(sequence_data: CustomerSequenceInput) -> SequencePredictionResponse:
    """Predict churn from customer sequence metrics using PyTorch LSTM capturing multi-period dependencies."""
    return ChurnPredictorService.predict_sequence(sequence_data, model_type="LSTM")


@router.post("/combined", response_model=CombinedPredictionResponse, summary="Combined ANN + LSTM Architecture")
async def predict_combined(payload: CombinedInput) -> CombinedPredictionResponse:
    """Advanced combined architecture: evaluates static profile (ANN) + temporal trajectory (LSTM)."""
    return ChurnPredictorService.predict_combined(payload)


@router.post("/batch", response_model=BatchPredictionSummary, summary="Batch Customer Churn Scoring")
async def predict_batch(batch_input: BatchCustomerInput) -> BatchPredictionSummary:
    """Score multiple customer records in a single batch and return aggregate retention portfolio analysis."""
    return ChurnPredictorService.predict_batch(batch_input.customers)


@router.post("/upload-csv", response_model=BatchPredictionSummary, summary="Score Customers via CSV File Upload")
async def score_csv_file(file: UploadFile = File(...)) -> BatchPredictionSummary:
    """Upload a CSV file of customer data (up to 500 rows preview), compute churn probabilities and retention plans."""
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    contents = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV file: {str(e)}")

    # Sample if large for responsive interactive preview
    if len(df) > 500:
        df = df.head(500)

    customer_list: List[CustomerInput] = []
    for _, row in df.iterrows():
        try:
            row_dict = row.to_dict()
            # Clean nan values
            cleaned = {k: (None if pd.isna(v) else v) for k, v in row_dict.items()}
            # Construct customer input with fallback to defaults
            cust = CustomerInput(
                customer_id=str(cleaned.get("customer_id", f"CUST{len(customer_list)+1}")),
                age=int(cleaned.get("age", 40) or 40),
                gender=str(cleaned.get("gender", "Female")),
                annual_income=float(cleaned.get("annual_income", 55000.0) or 55000.0),
                education=str(cleaned.get("education", "college")),
                marital_status=str(cleaned.get("marital_status", "married")),
                dependents=int(cleaned.get("dependents", 0) or 0),
                tenure=int(cleaned.get("tenure", 12) or 12),
                contract=str(cleaned.get("contract", "month-to-month")),
                payment_method=str(cleaned.get("payment_method", "electronic_check")),
                paperless_billing=str(cleaned.get("paperless_billing", "Yes")),
                senior_citizen=int(cleaned.get("senior_citizen", 0) or 0),
                monthlycharges=float(cleaned.get("monthlycharges", 70.0) or 70.0),
                totalcharges=float(cleaned.get("totalcharges", 800.0) or 800.0),
                num_services=int(cleaned.get("num_services", 2) or 2),
                has_phone_service=int(cleaned.get("has_phone_service", 1) or 1),
                has_internet_service=int(cleaned.get("has_internet_service", 1) or 1),
                customer_satisfaction=float(cleaned.get("customer_satisfaction", 6.0) or 6.0),
                num_complaints=float(cleaned.get("num_complaints", 0.0) or 0.0),
                num_service_calls=float(cleaned.get("num_service_calls", 1.0) or 1.0),
                late_payments=float(cleaned.get("late_payments", 0.0) or 0.0),
                avg_monthly_gb=float(cleaned.get("avg_monthly_gb", 45.0) or 45.0),
                days_since_last_interaction=float(cleaned.get("days_since_last_interaction", 14.0) or 14.0),
                credit_score=float(cleaned.get("credit_score", 650.0) or 650.0),
            )
            customer_list.append(cust)
        except Exception:
            continue

    if not customer_list:
        raise HTTPException(status_code=400, detail="Could not parse any valid customer rows from the uploaded file.")

    return ChurnPredictorService.predict_batch(customer_list)
