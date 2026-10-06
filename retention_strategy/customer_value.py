import pandas as pd


def calculate_value_thresholds(
    df,
    value_col="totalcharges",
    low_quantile=0.33,
    high_quantile=0.67
):
    """
    Calculate customer-value thresholds using quantiles.
    """

    if value_col not in df.columns:
        raise ValueError(f"Column '{value_col}' not found in dataframe.")

    low_threshold = df[value_col].quantile(low_quantile)
    high_threshold = df[value_col].quantile(high_quantile)

    return low_threshold, high_threshold


def assign_customer_value(
    value,
    low_threshold,
    high_threshold
):
    """
    Assign Low, Medium, or High customer value.
    """

    if value < low_threshold:
        return "Low"

    if value < high_threshold:
        return "Medium"

    return "High"


def add_customer_value(
    df,
    value_col="totalcharges",
    low_quantile=0.33,
    high_quantile=0.67
):
    """
    Add customer_value column to dataframe.
    """

    result = df.copy()

    low_threshold, high_threshold = calculate_value_thresholds(
        result,
        value_col=value_col,
        low_quantile=low_quantile,
        high_quantile=high_quantile
    )

    result["customer_value"] = result[value_col].apply(
        lambda value: assign_customer_value(
            value,
            low_threshold,
            high_threshold
        )
    )

    return result, {
        "low_threshold": low_threshold,
        "high_threshold": high_threshold
    }