import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from .config import (BATCH_SIZE_SEQ, COUNT_FEATURES, EVAL_BATCH_SIZE, SEED,
                     SEQ_FEATURES, SEQUENCE_LENGTH)


def load_and_split(csv_path):
    """Load the dataset and reproduce the notebook's 70/15/15 stratified split."""
    df = pd.read_csv(csv_path)
    df["signup_date"] = pd.to_datetime(df["signup_date"], errors="coerce")
    df = df.drop_duplicates()

    X = df.drop(columns=["customer_id", "signup_date", "churn"])
    y = df["churn"]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=SEED, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=SEED, stratify=y_temp
    )
    frames = {"train": X_train, "val": X_val, "test": X_test}
    labels = {
        "train": y_train.values.astype(np.float32),
        "val": y_val.values.astype(np.float32),
        "test": y_test.values.astype(np.float32),
    }
    return frames, labels


def build_behaviour_history(frame, medians, seq_len, seed):
    """SYNTHETIC monthly history, built WITHOUT churn labels.

    The final month always equals the customer's real observed record.
    """
    rng = np.random.default_rng(seed)
    final = frame[SEQ_FEATURES].fillna(medians).to_numpy(np.float32)   # (customers, features)
    n = len(final)
    seq = np.empty((n, seq_len, len(SEQ_FEATURES)), dtype=np.float32)

    for j, col in enumerate(SEQ_FEATURES):
        x = final[:, j]
        if col in COUNT_FEATURES:
            # Events spread randomly over the months; the cumulative count reaches the observed total.
            weights = rng.dirichlet(np.ones(seq_len), size=n)
            seq[:, :, j] = np.floor(np.cumsum(weights, axis=1) * x[:, None])
        elif col == "days_since_last_interaction":
            seq[:, :, j] = x[:, None] * rng.uniform(0.5, 1.5, (n, seq_len))
        else:
            # Usage / charges: a random walk that ends exactly at the observed value.
            steps = rng.normal(0, 0.05, (n, seq_len))
            steps[:, -1] = 0
            drift = np.cumsum(steps[:, ::-1], axis=1)[:, ::-1]
            seq[:, :, j] = x[:, None] * (1 + drift)
        seq[:, -1, j] = x

    return np.clip(seq, 0, None)


def build_sequences(frames, seq_len=SEQUENCE_LENGTH):
    """Build raw sequences per split and scale them with a train-only StandardScaler."""
    seq_medians = frames["train"][SEQ_FEATURES].median()      # train-only statistics
    seeds = {"train": 42, "val": 43, "test": 44}
    seq_raw = {
        split: build_behaviour_history(frame, seq_medians, seq_len, seed=seeds[split])
        for split, frame in frames.items()
    }

    n_feat = len(SEQ_FEATURES)
    seq_scaler = StandardScaler().fit(seq_raw["train"].reshape(-1, n_feat))   # fit on train only
    seq_data = {
        split: seq_scaler.transform(arr.reshape(-1, n_feat)).reshape(arr.shape).astype(np.float32)
        for split, arr in seq_raw.items()
    }
    return seq_data, seq_scaler


class CustomerSequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)                                   # (customers, months, features)
        self.y = torch.from_numpy(y.astype(np.float32)).unsqueeze(1)   # (customers, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def make_loaders(seq_data, seq_labels, batch_size=BATCH_SIZE_SEQ):
    return {
        split: DataLoader(
            CustomerSequenceDataset(seq_data[split], seq_labels[split]),
            batch_size=batch_size if split == "train" else EVAL_BATCH_SIZE,
            shuffle=(split == "train"),
        )
        for split in seq_data
    }
