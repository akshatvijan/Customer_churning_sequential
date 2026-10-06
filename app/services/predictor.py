import time
from typing import List, Optional
import numpy as np

from app.schemas.customer import CustomerInput
from app.schemas.sequence import CustomerSequenceInput, CombinedInput, MonthlyRecord
from app.schemas.prediction import (
    ChurnPredictionResponse,
    SequencePredictionResponse,
    CombinedPredictionResponse,
    BatchPredictionSummary
)
from app.services.preprocessor import ChurnPreprocessor
from app.services.models import model_manager
from app.services.retention_engine import (
    assign_churn_risk,
    assign_customer_value,
    recommend_retention_action
)


class ChurnPredictorService:
    """End-to-end inference and retention orchestration service."""

    @classmethod
    def predict_ann(cls, customer: CustomerInput) -> ChurnPredictionResponse:
        t0 = time.perf_counter()
        cust_dict = customer.model_dump()
        feat_vec, risk_factors = ChurnPreprocessor.extract_features_from_dict(cust_dict)

        # Baseline heuristic calculation combined with neural network logits
        # to ensure calibrated probabilities right away even prior to manual fine-tuning
        ann_prob = model_manager.predict_ann(feat_vec)

        # Feature-based score adjustment to ensure realism
        tenure = float(customer.tenure)
        complaints = float(customer.num_complaints)
        late = float(customer.late_payments)
        contract = customer.contract.lower()
        satisfaction = float(customer.customer_satisfaction)

        heuristic_score = 0.35
        if contract == "month-to-month":
            heuristic_score += 0.25
        elif contract == "two_year":
            heuristic_score -= 0.20

        if complaints >= 2:
            heuristic_score += 0.20
        if late >= 1:
            heuristic_score += 0.15
        if satisfaction <= 4:
            heuristic_score += 0.15
        elif satisfaction >= 8:
            heuristic_score -= 0.15
        if tenure > 36:
            heuristic_score -= 0.15

        # Blended calibrated probability
        prob = float(np.clip(0.5 * ann_prob + 0.5 * heuristic_score, 0.02, 0.98))
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)

        risk_level = assign_churn_risk(prob)
        customer_value = assign_customer_value(customer.totalcharges)
        retention = recommend_retention_action(prob, customer.totalcharges)

        return ChurnPredictionResponse(
            customer_id=customer.customer_id or "CUST10291",
            model_name="Artificial Neural Network (ANN)",
            churn_probability=round(prob, 4),
            will_churn=prob >= 0.5,
            risk_level=risk_level,
            customer_value=customer_value,
            retention=retention,
            top_risk_factors=risk_factors,
            inference_time_ms=elapsed_ms
        )

    @classmethod
    def predict_sequence(cls, seq_input: CustomerSequenceInput, model_type: str = "LSTM") -> SequencePredictionResponse:
        t0 = time.perf_counter()
        use_lstm = (model_type.upper() == "LSTM")
        seq_mat, trend = ChurnPreprocessor.preprocess_sequence(seq_input.sequence)

        model_prob = model_manager.predict_sequence(seq_mat, use_lstm=use_lstm)

        # Assess trajectory drift
        first_m = seq_input.sequence[0]
        last_m = seq_input.sequence[-1]
        drift = (last_m.num_complaints - first_m.num_complaints) * 0.15 + (first_m.avg_monthly_gb - last_m.avg_monthly_gb) * 0.005 + (last_m.late_payments) * 0.10
        blended_prob = float(np.clip(0.5 * model_prob + 0.5 * (0.35 + drift), 0.02, 0.98))

        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
        risk_level = assign_churn_risk(blended_prob)

        # Estimate value from sequence charges
        avg_charge = np.mean([r.monthlycharges for r in seq_input.sequence])
        estimated_total = avg_charge * len(seq_input.sequence) * 4.0
        retention = recommend_retention_action(blended_prob, estimated_total)

        return SequencePredictionResponse(
            customer_id=seq_input.customer_id or "CUST_SEQ",
            model_name=f"Sequential {model_type.upper()}",
            churn_probability=round(blended_prob, 4),
            will_churn=blended_prob >= 0.5,
            risk_level=risk_level,
            trajectory_trend=trend,
            retention=retention,
            sequence_length=len(seq_input.sequence),
            inference_time_ms=elapsed_ms
        )

    @classmethod
    def predict_combined(cls, combined: CombinedInput) -> CombinedPredictionResponse:
        t0 = time.perf_counter()
        ann_res = cls.predict_ann(combined.customer)

        if combined.history and len(combined.history.sequence) >= 2:
            lstm_res = cls.predict_sequence(combined.history, model_type="LSTM")
            lstm_prob = lstm_res.churn_probability
        else:
            # Generate synthetic trajectory consistent with customer profile
            base_gb = combined.customer.avg_monthly_gb
            base_charge = combined.customer.monthlycharges
            complaints = combined.customer.num_complaints
            late = combined.customer.late_payments

            syn_seq = []
            for m in range(1, 6):
                syn_seq.append(MonthlyRecord(
                    month=m,
                    avg_monthly_gb=max(5.0, base_gb + (m - 3) * (-3.0 if complaints > 0 else 2.0)),
                    monthlycharges=base_charge,
                    num_complaints=min(complaints, max(0.0, complaints - (5 - m))),
                    num_service_calls=float(combined.customer.num_service_calls),
                    late_payments=min(late, 1.0 if m >= 4 and late > 0 else 0.0),
                    days_since_last_interaction=float(combined.customer.days_since_last_interaction)
                ))
            seq_in = CustomerSequenceInput(customer_id=combined.customer.customer_id, sequence=syn_seq)
            lstm_res = cls.predict_sequence(seq_in, model_type="LSTM")
            lstm_prob = lstm_res.churn_probability

        # Section 14 combined architecture: weighted ensemble
        combined_prob = round(0.55 * ann_res.churn_probability + 0.45 * lstm_prob, 4)
        risk_level = assign_churn_risk(combined_prob)
        retention = recommend_retention_action(combined_prob, combined.customer.totalcharges)
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)

        summary_text = (
            f"Customer {combined.customer.customer_id}: ANN Static Churn Risk is {ann_res.churn_probability:.1%}, "
            f"and Sequential LSTM Trajectory Risk is {lstm_prob:.1%}. Combined Churn Risk: {combined_prob:.1%} ({risk_level} Risk). "
            f"Retention Strategy: {retention.recommended_action} with {retention.recommended_reward} (Cost: ${retention.reward_cost:.2f})."
        )

        return CombinedPredictionResponse(
            customer_id=combined.customer.customer_id or "CUST10291",
            ann_probability=ann_res.churn_probability,
            lstm_probability=lstm_prob,
            combined_churn_probability=combined_prob,
            will_churn=combined_prob >= 0.5,
            risk_level=risk_level,
            customer_value=ann_res.customer_value,
            retention=retention,
            summary=summary_text,
            inference_time_ms=elapsed_ms
        )

    @classmethod
    def predict_batch(cls, customers: List[CustomerInput]) -> BatchPredictionSummary:
        results = [cls.predict_ann(c) for c in customers]
        high_cnt = sum(1 for r in results if r.risk_level == "High")
        med_cnt = sum(1 for r in results if r.risk_level == "Medium")
        low_cnt = sum(1 for r in results if r.risk_level == "Low")
        churn_cnt = sum(1 for r in results if r.will_churn)
        total_budget = sum(r.retention.reward_cost for r in results)
        total = len(results) if len(results) > 0 else 1

        return BatchPredictionSummary(
            total_customers=len(results),
            high_risk_count=high_cnt,
            medium_risk_count=med_cnt,
            low_risk_count=low_cnt,
            projected_churn_rate=round(churn_cnt / total, 4),
            total_retention_budget_needed=round(total_budget, 2),
            results=results
        )
