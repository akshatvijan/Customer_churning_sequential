import joblib
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Any, Optional
from app.schemas.customer import CustomerInput
from app.schemas.sequence import MonthlyRecord
from app.config import SEQ_FEATURES, COUNT_FEATURES, SEQUENCE_LENGTH, SEQ_SCALER_PATH


# Categorical domain values based on dataset
EDUCATION_LEVELS = ["high_school", "college", "bachelor", "master", "phd"]
MARITAL_STATUSES = ["single", "married", "divorced", "widowed"]
CONTRACT_TYPES = ["month-to-month", "one_year", "two_year"]
PAYMENT_METHODS = ["electronic_check", "credit_card", "bank_transfer", "mailed_check"]
GENDERS = ["Female", "Male"]


class ChurnPreprocessor:
    """Preprocesses customer inputs and engineers temporal/behavioral features."""

    @staticmethod
    def extract_features_from_dict(d: dict) -> Tuple[np.ndarray, List[str]]:
        """Turn raw customer dictionary into engineered numerical vector and identify top risk factors."""
        tenure = max(1.0, float(d.get("tenure", 1.0)))
        totalcharges = float(d.get("totalcharges", 0.0))
        monthlycharges = max(1.0, float(d.get("monthlycharges", 50.0)))
        annual_income = float(d.get("annual_income", 50000.0))
        num_services = max(1.0, float(d.get("num_services", 1.0)))
        num_complaints = float(d.get("num_complaints", 0.0))
        num_service_calls = float(d.get("num_service_calls", 0.0))
        late_payments = float(d.get("late_payments", 0.0))
        avg_monthly_gb = float(d.get("avg_monthly_gb", 30.0))
        days_since_last_interaction = float(d.get("days_since_last_interaction", 10.0))
        customer_satisfaction = float(d.get("customer_satisfaction", 7.0))
        contract = str(d.get("contract", "month-to-month")).lower()
        payment_method = str(d.get("payment_method", "electronic_check")).lower()
        gender = str(d.get("gender", "Female"))
        education = str(d.get("education", "college")).lower()
        marital_status = str(d.get("marital_status", "married")).lower()
        paperless = 1.0 if str(d.get("paperless_billing", "Yes")).lower() in ["yes", "1", "true"] else 0.0
        senior = float(d.get("senior_citizen", 0))
        dependents = float(d.get("dependents", 0))
        age = float(d.get("age", 40))
        credit_score = float(d.get("credit_score", 650.0) or 650.0)

        # -----------------------------
        # Engineered Features (Kapil's Pipeline)
        # -----------------------------
        avg_charge_per_tenure = totalcharges / tenure
        support_burden = num_complaints + num_service_calls
        payment_risk = late_payments + num_complaints
        service_usage_ratio = num_services / tenure
        data_usage_per_service = avg_monthly_gb / num_services
        interaction_recency_score = days_since_last_interaction / 30.0
        monthly_income = annual_income / 12.0
        income_charge_ratio = monthly_income / monthlycharges

        # Service flags
        has_phone = float(d.get("has_phone_service", 1))
        has_internet = float(d.get("has_internet_service", 1))
        has_sec = float(d.get("has_online_security", 0))
        has_bkp = float(d.get("has_online_backup", 0))
        has_dev = float(d.get("has_device_protection", 0))
        has_tech = float(d.get("has_tech_support", 0))
        has_tv = float(d.get("has_streaming_tv", 0))
        has_movies = float(d.get("has_streaming_movies", 0))

        # Date parsing features
        signup_year = 2023.0
        signup_month = 6.0
        signup_quarter = 2.0
        signup_dayofweek = 2.0
        signup_str = str(d.get("signup_date", "2023-06-15"))
        try:
            dt = pd.to_datetime(signup_str)
            signup_year = float(dt.year)
            signup_month = float(dt.month)
            signup_quarter = float(dt.quarter)
            signup_dayofweek = float(dt.dayofweek)
        except Exception:
            pass

        # Normalized numericals
        num_vec = [
            (age - 45.0) / 15.0,
            (annual_income - 60000.0) / 30000.0,
            (tenure - 24.0) / 20.0,
            (monthlycharges - 70.0) / 30.0,
            (totalcharges - 2000.0) / 1800.0,
            (num_services - 3.0) / 2.0,
            (customer_satisfaction - 6.0) / 2.5,
            num_complaints / 3.0,
            num_service_calls / 3.0,
            late_payments / 2.0,
            (avg_monthly_gb - 50.0) / 30.0,
            (days_since_last_interaction - 15.0) / 15.0,
            (credit_score - 650.0) / 100.0,
            dependents / 2.0,
            senior,
            paperless,
            # Engineered
            (avg_charge_per_tenure - 70.0) / 40.0,
            support_burden / 4.0,
            payment_risk / 3.0,
            service_usage_ratio / 0.5,
            data_usage_per_service / 20.0,
            interaction_recency_score,
            (monthly_income - 5000.0) / 2500.0,
            income_charge_ratio / 50.0,
            (signup_year - 2022.0),
            (signup_month - 6.0) / 4.0,
            (signup_quarter - 2.5) / 1.5,
            (signup_dayofweek - 3.0) / 2.0,
            # Services
            has_phone, has_internet, has_sec, has_bkp, has_dev, has_tech, has_tv, has_movies
        ]

        # One-hot categorical encodings
        cat_vec = []
        for g in GENDERS:
            cat_vec.append(1.0 if gender.lower() == g.lower() else 0.0)
        for c in CONTRACT_TYPES:
            cat_vec.append(1.0 if contract == c else 0.0)
        for p in PAYMENT_METHODS:
            cat_vec.append(1.0 if payment_method == p else 0.0)
        for e in EDUCATION_LEVELS:
            cat_vec.append(1.0 if education == e else 0.0)
        for m in MARITAL_STATUSES:
            cat_vec.append(1.0 if marital_status == m else 0.0)

        feature_vector = np.array(num_vec + cat_vec, dtype=np.float32)

        # -----------------------------
        # Identify Top Risk Factors
        # -----------------------------
        risk_factors = []
        if contract == "month-to-month":
            risk_factors.append("Month-to-Month Contract (high churn correlation)")
        if num_complaints >= 2.0:
            risk_factors.append(f"High Complaint Volume ({int(num_complaints)} complaints filed)")
        if late_payments >= 1.0:
            risk_factors.append(f"Recorded Late Payments ({int(late_payments)} occurrences)")
        if customer_satisfaction <= 4.0:
            risk_factors.append(f"Low Satisfaction Rating ({customer_satisfaction}/10)")
        if num_service_calls >= 3.0:
            risk_factors.append(f"Frequent Support Inquiries ({int(num_service_calls)} calls)")
        if days_since_last_interaction > 30.0:
            risk_factors.append(f"Prolonged Inactivity ({int(days_since_last_interaction)} days without interaction)")
        if monthlycharges > 90.0 and income_charge_ratio < 40.0:
            risk_factors.append("High Cost Burden (Monthly bill high relative to income)")
        if tenure <= 6:
            risk_factors.append("Early Lifecycle Customer (Tenure <= 6 months)")

        if not risk_factors:
            risk_factors.append("Stable tenure and positive engagement metrics")

        return feature_vector, risk_factors

    _seq_scaler: Optional[Any] = None

    @classmethod
    def get_sequence_scaler(cls):
        """Lazy load and cache the sequence scaler saved by Aditya's rnn pipeline."""
        if cls._seq_scaler is None and SEQ_SCALER_PATH.exists():
            try:
                cls._seq_scaler = joblib.load(SEQ_SCALER_PATH)
            except Exception as e:
                print(f"Warning: Could not load saved sequence scaler from {SEQ_SCALER_PATH} ({e})")
        return cls._seq_scaler

    @staticmethod
    def _fallback_scale(raw_mat: np.ndarray) -> np.ndarray:
        """Standard fallback scaling when sequence_scaler.pkl is not yet trained."""
        scaled = np.zeros_like(raw_mat, dtype=np.float32)
        scaled[:, 0] = (raw_mat[:, 0] - 50.0) / 30.0
        scaled[:, 1] = (raw_mat[:, 1] - 70.0) / 30.0
        scaled[:, 2] = raw_mat[:, 2] / 2.0
        scaled[:, 3] = raw_mat[:, 3] / 2.0
        scaled[:, 4] = raw_mat[:, 4] / 2.0
        scaled[:, 5] = (raw_mat[:, 5] - 15.0) / 15.0
        return scaled

    @classmethod
    def preprocess_sequence(cls, records: List[MonthlyRecord]) -> Tuple[np.ndarray, str]:
        """Preprocess 5-month behavioural sequence for RNN / LSTM input.

        Uses Aditya's trained sequence_scaler.pkl when available, otherwise falls back to
        calibrated heuristic scaling.
        Output: numpy array of shape (1, seq_len, 6) and trajectory classification.
        """
        sorted_records = sorted(records, key=lambda r: r.month)
        seq_len = len(sorted_records)

        raw_mat = np.zeros((seq_len, len(SEQ_FEATURES)), dtype=np.float32)
        for i, rec in enumerate(sorted_records):
            raw_mat[i, 0] = float(rec.avg_monthly_gb)
            raw_mat[i, 1] = float(rec.monthlycharges)
            raw_mat[i, 2] = float(rec.num_complaints)
            raw_mat[i, 3] = float(rec.num_service_calls)
            raw_mat[i, 4] = float(rec.late_payments)
            raw_mat[i, 5] = float(rec.days_since_last_interaction)

        scaler = cls.get_sequence_scaler()
        if scaler is not None:
            try:
                scaled_mat = scaler.transform(raw_mat).astype(np.float32)
            except Exception as e:
                print(f"Warning: Sequence scaler transform failed ({e}), using fallback.")
                scaled_mat = cls._fallback_scale(raw_mat)
        else:
            scaled_mat = cls._fallback_scale(raw_mat)

        # Assess trajectory trend: comparing final month vs first month
        first_m = sorted_records[0]
        last_m = sorted_records[-1]

        complaint_growth = last_m.num_complaints - first_m.num_complaints
        usage_drop = first_m.avg_monthly_gb - last_m.avg_monthly_gb
        late_growth = last_m.late_payments - first_m.late_payments

        if complaint_growth > 0 or usage_drop > 15.0 or late_growth > 0:
            trend = "Deteriorating (Increasing risk indicators over time)"
        elif usage_drop < -10.0 and complaint_growth <= 0:
            trend = "Improving (Growing usage and zero complaints)"
        else:
            trend = "Stable (Consistent behavioural pattern)"

        return scaled_mat.reshape(1, seq_len, len(SEQ_FEATURES)), trend

    @classmethod
    def generate_synthetic_sequence(cls, customer: CustomerInput, seq_len: int = SEQUENCE_LENGTH, seed: int = 42) -> List[MonthlyRecord]:
        """Synthesize a multi-month behavioural trajectory from customer profile.

        Adopts Aditya's Dirichlet event accumulation and random-walk algorithm (from rnn/data.py),
        guaranteeing the final month strictly equals the customer's observed values.
        """
        rng = np.random.default_rng(seed)
        final_vals = np.array([
            customer.avg_monthly_gb,
            customer.monthlycharges,
            customer.num_complaints,
            customer.num_service_calls,
            customer.late_payments,
            customer.days_since_last_interaction,
        ], dtype=np.float32)

        seq = np.empty((seq_len, len(SEQ_FEATURES)), dtype=np.float32)

        for j, col in enumerate(SEQ_FEATURES):
            val = final_vals[j]
            if col in COUNT_FEATURES:
                weights = rng.dirichlet(np.ones(seq_len))
                seq[:, j] = np.floor(np.cumsum(weights) * val)
            elif col == "days_since_last_interaction":
                seq[:, j] = val * rng.uniform(0.5, 1.5, size=seq_len)
            else:
                steps = rng.normal(0, 0.05, size=seq_len)
                steps[-1] = 0.0
                drift = np.cumsum(steps[::-1])[::-1]
                seq[:, j] = val * (1.0 + drift)
            seq[-1, j] = val

        seq = np.clip(seq, 0.0, None)

        records: List[MonthlyRecord] = []
        for m in range(seq_len):
            records.append(MonthlyRecord(
                month=m + 1,
                avg_monthly_gb=round(float(seq[m, 0]), 1),
                monthlycharges=round(float(seq[m, 1]), 2),
                num_complaints=float(int(seq[m, 2])),
                num_service_calls=float(int(seq[m, 3])),
                late_payments=float(int(seq[m, 4])),
                days_since_last_interaction=round(float(seq[m, 5]), 1)
            ))
        return records
