import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score,
                             roc_auc_score)


def best_f1_threshold(y, p):
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec * rec / (prec + rec + 1e-12)
    return thr[np.argmax(f1[:-1])]


def evaluate(name, data_type, p_val, y_val, p_test, y_test, train_time, infer_time, n_params):
    thr = best_f1_threshold(y_val, p_val)      # threshold chosen on validation, applied to test
    pred = (p_test >= thr).astype(int)
    print(f"\n{name} confusion matrix (threshold={thr:.3f}):\n{confusion_matrix(y_test, pred)}")
    return {
        "Model": name, "Data Type": data_type,
        "ROC-AUC": roc_auc_score(y_test, p_test),
        "PR-AUC": average_precision_score(y_test, p_test),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred),
        "F1": f1_score(y_test, pred),
        "Training Time (s)": train_time,
        "Inference Time (s)": infer_time,
        "Parameters": n_params,
    }


def logreg_baselines(seq_data, seq_labels):
    """Baselines that answer "is the sequence justified?"."""
    baselines = {
        "LogReg - last month only": lambda X: X[:, -1, :],         # same 6 features, no history
        "LogReg - flattened history": lambda X: X.reshape(len(X), -1),
    }
    rows = []
    for name, view in baselines.items():
        t0 = time.perf_counter()
        lr_model = LogisticRegression(max_iter=1000).fit(view(seq_data["train"]), seq_labels["train"])
        t_train = time.perf_counter() - t0
        p_val = lr_model.predict_proba(view(seq_data["val"]))[:, 1]
        t0 = time.perf_counter()
        p_test = lr_model.predict_proba(view(seq_data["test"]))[:, 1]
        t_inf = time.perf_counter() - t0
        rows.append(evaluate(name, "Tabular", p_val, seq_labels["val"], p_test, seq_labels["test"],
                             t_train, t_inf, lr_model.coef_.size + 1))
    return rows
