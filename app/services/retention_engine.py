from typing import Dict, Tuple, Optional
from app.config import (
    DEFAULT_RETENTION_RULES,
    LOW_RISK_THRESHOLD,
    HIGH_RISK_THRESHOLD,
    LOW_VALUE_THRESHOLD,
    HIGH_VALUE_THRESHOLD,
)
from app.schemas.retention import RetentionRecommendation, ROIResponse


def assign_churn_risk(
    churn_probability: float,
    low_threshold: float = LOW_RISK_THRESHOLD,
    high_threshold: float = HIGH_RISK_THRESHOLD,
) -> str:
    """Assign Low, Medium, or High churn risk based on model probability."""
    if not (0.0 <= churn_probability <= 1.0):
        churn_probability = max(0.0, min(1.0, churn_probability))

    if churn_probability < low_threshold:
        return "Low"
    elif churn_probability < high_threshold:
        return "Medium"
    else:
        return "High"


def assign_customer_value(
    totalcharges: float,
    low_threshold: float = LOW_VALUE_THRESHOLD,
    high_threshold: float = HIGH_VALUE_THRESHOLD,
) -> str:
    """Segment customer into Low, Medium, or High value tier based on cumulative spend."""
    if totalcharges < low_threshold:
        return "Low"
    elif totalcharges < high_threshold:
        return "Medium"
    else:
        return "High"


def get_retention_rule(
    risk_level: str,
    customer_value: str,
    rules: Optional[Dict[Tuple[str, str], dict]] = None,
) -> dict:
    """Lookup rule configuration from matrix."""
    rule_table = rules if rules is not None else DEFAULT_RETENTION_RULES
    key = (risk_level, customer_value)
    if key in rule_table:
        return rule_table[key]
    # Fallback default
    return {
        "action": "MONITOR",
        "reward": "No reward",
        "reward_cost": 0.0,
        "description": "Standard account monitoring."
    }


def recommend_retention_action(
    churn_probability: float,
    totalcharges: float,
    low_risk_threshold: float = LOW_RISK_THRESHOLD,
    high_risk_threshold: float = HIGH_RISK_THRESHOLD,
    low_value_threshold: float = LOW_VALUE_THRESHOLD,
    high_value_threshold: float = HIGH_VALUE_THRESHOLD,
    rules: Optional[Dict[Tuple[str, str], dict]] = None,
) -> RetentionRecommendation:
    """Generate comprehensive retention recommendation and budget allocation."""
    risk_level = assign_churn_risk(churn_probability, low_risk_threshold, high_risk_threshold)
    customer_value = assign_customer_value(totalcharges, low_value_threshold, high_value_threshold)

    rule = get_retention_rule(risk_level, customer_value, rules)

    # For low churn probability customers, avoid incurring unnecessary retention incentive costs
    if churn_probability < low_risk_threshold:
        return RetentionRecommendation(
            risk_level=risk_level,
            customer_value=customer_value,
            recommended_action="NO_INTERVENTION",
            recommended_reward="No reward",
            reward_cost=0.0,
            description="Healthy retention indicators; standard organic lifecycle engagement."
        )

    return RetentionRecommendation(
        risk_level=risk_level,
        customer_value=customer_value,
        recommended_action=rule["action"],
        recommended_reward=rule["reward"],
        reward_cost=float(rule["reward_cost"]),
        description=rule.get("description", "Tailored retention action based on business matrix.")
    )


def calculate_roi(
    customers_targeted: int,
    expected_incremental_retention: float,
    average_customer_value: float,
    total_reward_cost: float,
    campaign_cost: float
) -> ROIResponse:
    """Calculate campaign net business impact and projected Return on Investment (ROI)."""
    potential_retained_value = (
        customers_targeted * expected_incremental_retention * average_customer_value
    )
    total_campaign_cost = total_reward_cost + campaign_cost
    net_business_impact = potential_retained_value - total_campaign_cost

    if total_campaign_cost > 0:
        roi_percentage = (net_business_impact / total_campaign_cost) * 100.0
    else:
        roi_percentage = 0.0

    return ROIResponse(
        customers_targeted=customers_targeted,
        expected_incremental_retention=expected_incremental_retention,
        average_customer_value=average_customer_value,
        potential_retained_value=round(potential_retained_value, 2),
        total_reward_cost=round(total_reward_cost, 2),
        campaign_cost=round(campaign_cost, 2),
        total_campaign_cost=round(total_campaign_cost, 2),
        net_business_impact=round(net_business_impact, 2),
        roi_percentage=round(roi_percentage, 2)
    )
