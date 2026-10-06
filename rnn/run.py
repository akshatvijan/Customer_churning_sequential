"""Train and compare RNN / LSTM churn models.

Usage:
    python -m rnn.run --data dataset/customer_churn_1M.csv
"""
import argparse
import time

import numpy as np
import pandas as pd
import torch

from .config import DATA_PATH, SEED, SEQ_FEATURES, SEQ_MODEL_DIR, get_device
from .data import build_sequences, load_and_split, make_loaders
from .evaluate import evaluate, logreg_baselines
from .models import ChurnLSTM, ChurnRNN
from .train import predict_seq, train_sequence_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(DATA_PATH))
    parser.add_argument("--model-dir", default=str(SEQ_MODEL_DIR))
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--plot", action="store_true", help="Show learning curves")
    args = parser.parse_args()

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = get_device()
    print("Using device:", device)

    frames, seq_labels = load_and_split(args.data)
    seq_data, seq_scaler = build_sequences(frames)
    loaders = make_loaders(seq_data, seq_labels)

    xb, _ = next(iter(loaders["train"]))
    print("Batch input shape (batch_size, sequence_length, number_of_features):", tuple(xb.shape))

    n_feat = len(SEQ_FEATURES)
    common = dict(loaders=loaders, y_train=seq_labels["train"], device=device,
                  seq_scaler=seq_scaler, model_dir=args.model_dir, epochs=args.epochs)

    torch.manual_seed(SEED)
    rnn_model = ChurnRNN(n_feat, hidden_size=64).to(device)
    rnn_history, rnn_train_time = train_sequence_model(rnn_model, "RNN", **common)

    torch.manual_seed(SEED)
    lstm_model = ChurnLSTM(n_feat, hidden_size=64, num_layers=1).to(device)
    lstm_history, lstm_train_time = train_sequence_model(lstm_model, "LSTM", **common)

    rows = []
    for name, model_, t_train in [("RNN", rnn_model, rnn_train_time), ("LSTM", lstm_model, lstm_train_time)]:
        p_val, y_val = predict_seq(model_, loaders["val"], device)
        t0 = time.perf_counter()
        p_test, y_test = predict_seq(model_, loaders["test"], device)
        t_inf = time.perf_counter() - t0
        rows.append(evaluate(name, "Sequential (synthetic)", p_val, y_val, p_test, y_test,
                             t_train, t_inf, sum(p.numel() for p in model_.parameters())))

    rows += logreg_baselines(seq_data, seq_labels)

    comparison_df = pd.DataFrame(rows).set_index("Model")
    print("\n", comparison_df.round(4).to_string())

    if args.plot:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for name, hist in [("RNN", rnn_history), ("LSTM", lstm_history)]:
            axes[0].plot(hist["train_loss"], label=name)
            axes[1].plot(hist["val_pr_auc"], label=name)
        axes[0].set_title("Train loss")
        axes[1].set_title("Validation PR-AUC")
        for ax in axes:
            ax.set_xlabel("Epoch")
            ax.legend()
        plt.show()


if __name__ == "__main__":
    main()
