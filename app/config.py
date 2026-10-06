import os
from pathlib import Path
from typing import Dict, Tuple

BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"

# Ensure directories exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Model file paths
ANN_MODEL_PATH = MODELS_DIR / "churn_ann.pt"
RNN_MODEL_PATH = MODELS_DIR / "churn_rnn.pt"
LSTM_MODEL_PATH = MODELS_DIR / "churn_lstm.pt"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.joblib"

# Churn Risk Boundaries (matching Ridhima's retention strategy)
LOW_RISK_THRESHOLD = 0.40
HIGH_RISK_THRESHOLD = 0.70

# Customer Value Thresholds (total charges boundaries)
LOW_VALUE_THRESHOLD = 1500.0
HIGH_VALUE_THRESHOLD = 4000.0

# Sequential features for RNN / LSTM (matching Aditya's rnn/config.py)
SEQ_FEATURES = [
    "avg_monthly_gb",
    "monthlycharges",
    "num_complaints",
    "num_service_calls",
    "late_payments",
    "days_since_last_interaction",
]
SEQUENCE_LENGTH = 5

# Retention Strategy Business Rule Table
DEFAULT_RETENTION_RULES: Dict[Tuple[str, str], dict] = {
    ("Low", "Low"): {
        "action": "NO_INTERVENTION",
        "reward": "No reward",
        "reward_cost": 0.0,
        "description": "Standard service with automated self-service engagement."
    },
    ("Low", "Medium"): {
        "action": "LOYALTY_BENEFIT",
        "reward": "Loyalty points",
        "reward_cost": 5.0,
        "description": "Reward loyalty to maintain brand affinity."
    },
    ("Low", "High"): {
        "action": "LOYALTY_BENEFIT",
        "reward": "Premium loyalty benefit",
        "reward_cost": 10.0,
        "description": "Deliver high-value perks for top-tier low-churn customers."
    },
    ("Medium", "Low"): {
        "action": "MONITOR",
        "reward": "No reward",
        "reward_cost": 0.0,
        "description": "Flag for tracking; avoid expensive spend on low-value medium risk."
    },
    ("Medium", "Medium"): {
        "action": "PERSONALIZED_OFFER",
        "reward": "Personalized discount",
        "reward_cost": 15.0,
        "description": "Offer tailored 10-15% discount or bundled upgrade."
    },
    ("Medium", "High"): {
        "action": "PERSONALIZED_REWARD",
        "reward": "Personalized reward",
        "reward_cost": 25.0,
        "description": "High-touch engagement with dedicated account manager review."
    },
    ("High", "Low"): {
        "action": "LOW_COST_RETENTION",
        "reward": "Low-cost retention offer",
        "reward_cost": 10.0,
        "description": "Automated digital promo code or retention coupon."
    },
    ("High", "Medium"): {
        "action": "RETENTION_REWARD",
        "reward": "Moderate discount",
        "reward_cost": 25.0,
        "description": "20% off renewal and proactive customer support outreach."
    },
    ("High", "High"): {
        "action": "PRIORITY_RETENTION",
        "reward": "Premium retention reward",
        "reward_cost": 50.0,
        "description": "Immediate white-glove executive intervention & customized renewal discount."
    }
}
