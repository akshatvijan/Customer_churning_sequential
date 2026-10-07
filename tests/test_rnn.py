import pytest
import torch
import numpy as np
import pandas as pd

from rnn.models import ChurnRNN, ChurnLSTM
from rnn.config import SEQ_FEATURES, SEQUENCE_LENGTH, COUNT_FEATURES
from rnn.data import build_behaviour_history, CustomerSequenceDataset, make_loaders
from rnn.evaluate import best_f1_threshold, evaluate
from app.services.models import model_manager
from app.services.preprocessor import ChurnPreprocessor
from app.schemas.customer import CustomerInput
from app.schemas.sequence import CustomerSequenceInput, MonthlyRecord
from app.services.predictor import ChurnPredictorService


def test_rnn_and_lstm_architectures():
    batch_size = 4
    seq_len = 5
    n_feat = len(SEQ_FEATURES)

    x = torch.randn(batch_size, seq_len, n_feat)

    rnn = ChurnRNN(input_size=n_feat, hidden_size=64, num_layers=1)
    rnn_out = rnn(x)
    assert rnn_out.shape == (batch_size, 1)

    lstm = ChurnLSTM(input_size=n_feat, hidden_size=64, num_layers=1, dropout=0.2)
    lstm_out = lstm(x)
    assert lstm_out.shape == (batch_size, 1)


def test_synthetic_behaviour_history():
    df = pd.DataFrame({
        "avg_monthly_gb": [50.0, 75.0],
        "monthlycharges": [70.0, 85.0],
        "num_complaints": [2.0, 0.0],
        "num_service_calls": [3.0, 1.0],
        "late_payments": [1.0, 0.0],
        "days_since_last_interaction": [15.0, 5.0],
    })
    medians = df.median()
    seq = build_behaviour_history(df, medians=medians, seq_len=5, seed=42)

    assert seq.shape == (2, 5, 6)
    assert np.all(seq >= 0.0)

    # Final month should match the observed values
    np.testing.assert_allclose(seq[0, -1, :], df.iloc[0].to_numpy(np.float32), rtol=1e-5)
    np.testing.assert_allclose(seq[1, -1, :], df.iloc[1].to_numpy(np.float32), rtol=1e-5)


def test_sequence_dataset_and_loader():
    X = np.random.randn(20, 5, 6).astype(np.float32)
    y = np.random.randint(0, 2, size=(20,)).astype(np.float32)

    dataset = CustomerSequenceDataset(X, y)
    assert len(dataset) == 20
    xb, yb = dataset[0]
    assert xb.shape == (5, 6)
    assert yb.shape == (1,)

    loaders = make_loaders({"train": X, "val": X}, {"train": y, "val": y}, batch_size=8)
    assert "train" in loaders
    xb_batch, yb_batch = next(iter(loaders["train"]))
    assert xb_batch.shape[0] <= 8
    assert xb_batch.shape[1:] == (5, 6)


def test_evaluate_and_threshold():
    y = np.array([0, 0, 1, 1], dtype=np.float32)
    p = np.array([0.1, 0.4, 0.8, 0.9], dtype=np.float32)

    thr = best_f1_threshold(y, p)
    assert 0.0 <= thr <= 1.0

    res = evaluate("TestModel", "Sequential", p, y, p, y, train_time=1.0, infer_time=0.01, n_params=5000)
    assert res["Model"] == "TestModel"
    assert res["ROC-AUC"] >= 0.9
    assert res["F1"] >= 0.9


def test_preprocessor_synthetic_sequence():
    cust = CustomerInput(
        customer_id="CUST_SYN_TEST",
        avg_monthly_gb=60.0,
        monthlycharges=80.0,
        num_complaints=2.0,
        num_service_calls=3.0,
        late_payments=1.0,
        days_since_last_interaction=20.0
    )
    records = ChurnPreprocessor.generate_synthetic_sequence(cust, seq_len=5, seed=42)
    assert len(records) == 5
    assert records[-1].month == 5
    assert records[-1].avg_monthly_gb == 60.0
    assert records[-1].monthlycharges == 80.0
    assert records[-1].num_complaints == 2.0
    assert records[-1].num_service_calls == 3.0
    assert records[-1].late_payments == 1.0
    assert records[-1].days_since_last_interaction == 20.0


def test_model_manager_checkpoint_dict_handling(tmp_path):
    # Test loading a checkpoint saved in Aditya's dictionary format
    model = ChurnRNN(input_size=6, hidden_size=64, num_layers=1)
    ckpt_path = tmp_path / "test_rnn_churn.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "seq_features": SEQ_FEATURES,
        "sequence_length": SEQUENCE_LENGTH,
        "best_validation_pr_auc": 0.725
    }, ckpt_path)

    # Test loading dict checkpoint
    ckpt = torch.load(ckpt_path, weights_only=False)
    assert "model_state_dict" in ckpt
    new_model = ChurnRNN(input_size=6, hidden_size=64, num_layers=1)
    new_model.load_state_dict(ckpt["model_state_dict"])
    assert ckpt["best_validation_pr_auc"] == 0.725
