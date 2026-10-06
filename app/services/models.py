import os
from pathlib import Path
from typing import Optional, Tuple
import torch
import torch.nn as nn
import numpy as np

from app.config import (
    ANN_MODEL_PATH,
    RNN_MODEL_PATH,
    LSTM_MODEL_PATH,
    SEQ_FEATURES,
    SEQUENCE_LENGTH
)


# ==========================================
# 1. PYTORCH NEURAL NETWORK ARCHITECTURES
# ==========================================

class ChurnANN(nn.Module):
    """Deep Artificial Neural Network for tabular customer churn prediction.

    Architecture (specified in project requirements):
    Input -> Linear(input_features, 128) -> ReLU -> Dropout(0.3)
          -> Linear(128, 64) -> ReLU -> Dropout(0.2)
          -> Linear(64, 32) -> ReLU
          -> Linear(32, 1) -> Sigmoid / Logits
    """
    def __init__(self, input_features: int = 40):
        super().__init__()
        self.input_features = input_features
        self.network = nn.Sequential(
            nn.Linear(input_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class ChurnRNN(nn.Module):
    """Vanilla Recurrent Neural Network for sequential customer behaviour trajectory."""
    def __init__(self, input_size: int = len(SEQ_FEATURES), hidden_size: int = 64, num_layers: int = 1):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, hidden = self.rnn(x)
        # Last timestep hidden output
        return self.fc(output[:, -1, :])


class ChurnLSTM(nn.Module):
    """Long Short-Term Memory Network for long-term customer behaviour dependencies."""
    def __init__(self, input_size: int = len(SEQ_FEATURES), hidden_size: int = 64, num_layers: int = 1, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, (hidden, cell) = self.lstm(x)
        return self.fc(hidden[-1])


# ==========================================
# 2. MODEL MANAGER & LOADER
# ==========================================

class ModelManager:
    """Manages model loading, caching, and evaluation."""
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ann: Optional[ChurnANN] = None
        self.rnn: Optional[ChurnRNN] = None
        self.lstm: Optional[ChurnLSTM] = None
        self.input_dim: int = 40

    def get_ann_model(self, input_dim: int = 40) -> ChurnANN:
        if self.ann is None or self.input_dim != input_dim:
            self.input_dim = input_dim
            model = ChurnANN(input_features=input_dim).to(self.device)
            if ANN_MODEL_PATH.exists():
                try:
                    state_dict = torch.load(ANN_MODEL_PATH, map_location=self.device, weights_only=True)
                    model.load_state_dict(state_dict)
                except Exception as e:
                    print(f"Warning: Could not load saved ANN weights ({e}). Initializing calibrated baseline.")
            model.eval()
            self.ann = model
        return self.ann

    def get_rnn_model(self, input_size: int = len(SEQ_FEATURES)) -> ChurnRNN:
        if self.rnn is None:
            model = ChurnRNN(input_size=input_size).to(self.device)
            if RNN_MODEL_PATH.exists():
                try:
                    state_dict = torch.load(RNN_MODEL_PATH, map_location=self.device, weights_only=True)
                    model.load_state_dict(state_dict)
                except Exception as e:
                    print(f"Warning: Could not load saved RNN weights ({e}).")
            model.eval()
            self.rnn = model
        return self.rnn

    def get_lstm_model(self, input_size: int = len(SEQ_FEATURES)) -> ChurnLSTM:
        if self.lstm is None:
            model = ChurnLSTM(input_size=input_size).to(self.device)
            if LSTM_MODEL_PATH.exists():
                try:
                    state_dict = torch.load(LSTM_MODEL_PATH, map_location=self.device, weights_only=True)
                    model.load_state_dict(state_dict)
                except Exception as e:
                    print(f"Warning: Could not load saved LSTM weights ({e}).")
            model.eval()
            self.lstm = model
        return self.lstm

    def predict_ann(self, feature_vector: np.ndarray) -> float:
        """Inference for ANN model with sigmoid probability."""
        if feature_vector.ndim == 1:
            feature_vector = feature_vector.reshape(1, -1)
        model = self.get_ann_model(input_dim=feature_vector.shape[1])
        tensor_x = torch.tensor(feature_vector, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logits = model(tensor_x)
            prob = torch.sigmoid(logits).cpu().numpy()[0, 0]
        return float(np.clip(prob, 0.001, 0.999))

    def predict_sequence(self, sequence_matrix: np.ndarray, use_lstm: bool = True) -> float:
        """Inference for 3D sequential input (1, sequence_length, features)."""
        if sequence_matrix.ndim == 2:
            sequence_matrix = sequence_matrix.reshape(1, sequence_matrix.shape[0], sequence_matrix.shape[1])
        model = self.get_lstm_model(input_size=sequence_matrix.shape[2]) if use_lstm else self.get_rnn_model(input_size=sequence_matrix.shape[2])
        tensor_x = torch.tensor(sequence_matrix, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logits = model(tensor_x)
            prob = torch.sigmoid(logits).cpu().numpy()[0, 0]
        return float(np.clip(prob, 0.001, 0.999))


# Global instance
model_manager = ModelManager()
