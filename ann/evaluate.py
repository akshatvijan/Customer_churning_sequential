import numpy as np
import torch

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)


def predict_probabilities(model, loader, device):

    model.eval()

    probabilities = []
    actuals = []

    with torch.no_grad():

        for X_batch, y_batch in loader:

            X_batch = X_batch.to(device)

            logits = model(X_batch)

            probabilities.extend(
                torch.sigmoid(logits)
                .cpu()
                .numpy()
                .ravel()
            )

            actuals.extend(
                y_batch.numpy().ravel()
            )

    return (
        np.array(probabilities),
        np.array(actuals)
    )


def evaluate_model(
    actuals,
    probabilities,
    threshold=0.5
):

    predictions = (
        probabilities >= threshold
    ).astype(int)

    metrics = {
        "accuracy": accuracy_score(
            actuals, predictions
        ),

        "precision": precision_score(
            actuals,
            predictions,
            zero_division=0
        ),

        "recall": recall_score(
            actuals,
            predictions,
            zero_division=0
        ),

        "f1": f1_score(
            actuals,
            predictions,
            zero_division=0
        ),

        "roc_auc": roc_auc_score(
            actuals,
            probabilities
        ),

        "pr_auc": average_precision_score(
            actuals,
            probabilities
        )
    }

    return metrics, confusion_matrix(
        actuals,
        predictions
    )