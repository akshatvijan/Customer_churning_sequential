import pandas as pd

from .risk import assign_churn_risk
from .customer_value import add_customer_value
from .reward_recommendations import recommend_retention_action


def run_retention_strategy(
    df,
    probability_col="churn_probability",
    value_col="totalcharges",
    rules=None
):
    """
    Run the complete retention strategy pipeline.

    Input:
        Customer dataframe containing churn probability
        and customer value information.

    Output:
        Customer dataframe containing:
        - churn risk
        - customer value
        - recommended action
        - recommended reward
        - reward cost
    """

    result = df.copy()

    # -----------------------------
    # 1. Churn Risk
    # -----------------------------

    result["risk_level"] = result[probability_col].apply(
        assign_churn_risk
    )

    # -----------------------------
    # 2. Customer Value
    # -----------------------------

    result, value_thresholds = add_customer_value(
        result,
        value_col=value_col
    )

    # -----------------------------
    # 3. Retention Recommendation
    # -----------------------------

    recommendations = result.apply(
        lambda row: recommend_retention_action(
            churn_probability=row[probability_col],
            risk_level=row["risk_level"],
            customer_value=row["customer_value"],
            rules=rules
        ),
        axis=1
    )

    recommendations_df = pd.DataFrame(
        recommendations.tolist(),
        index=result.index
    )

    result = pd.concat(
        [result, recommendations_df],
        axis=1
    )

    return result, value_thresholds