def assign_churn_risk(
    churn_probability,
    low_threshold=0.40,
    high_threshold=0.70
):
    """
    Assign a customer to Low, Medium, or High churn risk.

    Parameters
    ----------
    churn_probability : float
        Predicted probability of churn between 0 and 1.

    low_threshold : float
        Upper boundary for Low risk.

    high_threshold : float
        Lower boundary for High risk.

    Returns
    -------
    str
        Low, Medium, or High.
    """

    if not 0 <= churn_probability <= 1:
        raise ValueError("Churn probability must be between 0 and 1.")

    if churn_probability < low_threshold:
        return "Low"

    if churn_probability < high_threshold:
        return "Medium"

    return "High"