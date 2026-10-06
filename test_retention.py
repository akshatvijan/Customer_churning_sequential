import pandas as pd

from retention_strategy import (
    run_retention_strategy,
    calculate_roi
)


customers = pd.DataFrame({
    "customer_id": [101, 102, 103, 104, 105],
    "churn_probability": [0.85, 0.65, 0.25, 0.75, 0.50],
    "totalcharges": [5000, 2500, 800, 4500, 1800]
})


# Run retention strategy
result, thresholds = run_retention_strategy(
    customers
)

print("\nRetention Strategy Results:")
print(
    result[
        [
            "customer_id",
            "churn_probability",
            "risk_level",
            "customer_value",
            "recommended_action",
            "recommended_reward",
            "reward_cost"
        ]
    ]
)


# ROI calculation
targeted_customers = result[
    result["risk_level"] == "High"
]

roi = calculate_roi(
    customers_targeted=len(targeted_customers),
    expected_incremental_retention=0.20,
    average_customer_value=3000,
    total_reward_cost=targeted_customers["reward_cost"].sum(),
    campaign_cost=100
)

print("\nROI / Business Impact:")
for key, value in roi.items():
    print(f"{key}: {value}")