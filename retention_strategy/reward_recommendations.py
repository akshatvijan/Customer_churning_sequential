from .retention_rules import get_retention_rule


def recommend_retention_action(
    churn_probability,
    risk_level,
    customer_value,
    rules=None
):
    """
    Generate a retention recommendation
    based on churn risk and customer value.
    """

    rule = get_retention_rule(
        risk_level=risk_level,
        customer_value=customer_value,
        rules=rules
    )

    # Avoid expensive rewards for low churn probability.
    if churn_probability < 0.40:
        return {
            "recommended_action": "NO_INTERVENTION",
            "recommended_reward": "No reward",
            "reward_cost": 0.0
        }

    return {
        "recommended_action": rule["action"],
        "recommended_reward": rule["reward"],
        "reward_cost": rule["reward_cost"]
    }