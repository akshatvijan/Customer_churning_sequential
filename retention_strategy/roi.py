def calculate_roi(
    customers_targeted,
    expected_incremental_retention,
    average_customer_value,
    total_reward_cost,
    campaign_cost
):
    """
    Estimate retention campaign business impact and ROI.

    These are scenario-based estimates, not causal estimates.
    """

    potential_retained_value = (
        customers_targeted
        * expected_incremental_retention
        * average_customer_value
    )

    net_business_impact = (
        potential_retained_value
        - total_reward_cost
        - campaign_cost
    )

    total_campaign_cost = total_reward_cost + campaign_cost

    if total_campaign_cost > 0:
        roi_percentage = (
            net_business_impact / total_campaign_cost
        ) * 100
    else:
        roi_percentage = 0.0

    return {
        "customers_targeted": customers_targeted,
        "expected_incremental_retention":
            expected_incremental_retention,
        "average_customer_value":
            average_customer_value,
        "potential_retained_value":
            potential_retained_value,
        "total_reward_cost":
            total_reward_cost,
        "campaign_cost":
            campaign_cost,
        "net_business_impact":
            net_business_impact,
        "roi_percentage":
            roi_percentage
    }