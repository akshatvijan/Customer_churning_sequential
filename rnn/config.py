from pathlib import Path

import torch

SEED = 42

SEQ_FEATURES = [
    "avg_monthly_gb",               # usage
    "monthlycharges",               # charges
    "num_complaints",
    "num_service_calls",
    "late_payments",
    "days_since_last_interaction",
]
COUNT_FEATURES = ["num_complaints", "num_service_calls", "late_payments"]  # events that accumulate
SEQUENCE_LENGTH = 5   # doc example input: (64, 5, 6)

BATCH_SIZE_SEQ = 64
EVAL_BATCH_SIZE = 2048

DATA_PATH = Path("dataset/customer_churn_1M.csv")
SEQ_MODEL_DIR = Path("artifacts/models")


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
