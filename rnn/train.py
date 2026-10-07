import copy
import time
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score

from .config import SEQ_FEATURES, SEQUENCE_LENGTH


@torch.inference_mode()
def predict_seq(model, loader, device):
    model.eval()
    probs, ys = [], []
    for xb, yb in loader:
        probs.append(torch.sigmoid(model(xb.to(device))).cpu().numpy().ravel())
        ys.append(yb.numpy().ravel())
    return np.concatenate(probs), np.concatenate(ys)


def train_sequence_model(model, name, loaders, y_train, device, seq_scaler, model_dir,
                         epochs=15, patience=4, lr=1e-3):
    pos_weight = torch.tensor([(len(y_train) - y_train.sum()) / y_train.sum()], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=1)

    history = {"train_loss": [], "val_roc_auc": [], "val_pr_auc": []}
    best_pr, best_state, wait = -np.inf, None, 0
    start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for xb, yb in loaders["train"]:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # guards against exploding gradients
            optimizer.step()
            running += loss.item() * len(xb)

        p_val, y_val = predict_seq(model, loaders["val"], device)
        roc, pr = roc_auc_score(y_val, p_val), average_precision_score(y_val, p_val)
        scheduler.step(pr)

        history["train_loss"].append(running / len(loaders["train"].dataset))
        history["val_roc_auc"].append(roc)
        history["val_pr_auc"].append(pr)
        print(f"[{name}] Epoch {epoch:02d} | Train Loss {history['train_loss'][-1]:.4f} | "
              f"Val ROC-AUC {roc:.4f} | Val PR-AUC {pr:.4f} | LR {optimizer.param_groups[0]['lr']:.6f}")

        if pr > best_pr:
            best_pr, best_state, wait = pr, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= patience:
                print(f"[{name}] Early stopping at epoch {epoch}.")
                break

    train_time = time.perf_counter() - start
    model.load_state_dict(best_state)

    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "seq_features": SEQ_FEATURES,
        "sequence_length": SEQUENCE_LENGTH,
        "best_validation_pr_auc": best_pr,
    }, model_dir / f"{name.lower()}_churn.pt")
    joblib.dump(seq_scaler, model_dir / "sequence_scaler.pkl")
    return history, train_time
