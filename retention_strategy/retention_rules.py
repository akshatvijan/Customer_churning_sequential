DEFAULT_RETENTION_RULES = {
    ("Low", "Low"): {
        "action": "NO_INTERVENTION",
        "reward": "No reward",
        "reward_cost": 0.0
    },

    ("Low", "Medium"): {
        "action": "LOYALTY_BENEFIT",
        "reward": "Loyalty points",
        "reward_cost": 5.0
    },

    ("Low", "High"): {
        "action": "LOYALTY_BENEFIT",
        "reward": "Premium loyalty benefit",
        "reward_cost": 10.0
    },

    ("Medium", "Low"): {
        "action": "MONITOR",
        "reward": "No reward",
        "reward_cost": 0.0
    },

    ("Medium", "Medium"): {
        "action": "PERSONALIZED_OFFER",
        "reward": "Personalized discount",
        "reward_cost": 15.0
    },

    ("Medium", "High"): {
        "action": "PERSONALIZED_REWARD",
        "reward": "Personalized reward",
        "reward_cost": 25.0
    },

    ("High", "Low"): {
        "action": "LOW_COST_RETENTION",
        "reward": "Low-cost retention offer",
        "reward_cost": 10.0
    },

    ("High", "Medium"): {
        "action": "RETENTION_REWARD",
        "reward": "Moderate discount",
        "reward_cost": 25.0
    },

    ("High", "High"): {
        "action": "PRIORITY_RETENTION",
        "reward": "Premium retention reward",
        "reward_cost": 50.0
    }
}


def get_retention_rule(
    risk_level,
    customer_value,
    rules=None
):
    """
    Return the configured retention rule
    for a risk-value combination.
    """

    if rules is None:
        rules = DEFAULT_RETENTION_RULES

    key = (risk_level, customer_value)

    if key not in rules:
        raise ValueError(
            f"No retention rule configured for {risk_level}, "
            f"{customer_value}"
        )

    return rules[key]