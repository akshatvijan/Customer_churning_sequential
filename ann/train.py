import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import ChurnDataset
from model import ChurnANN


def train_one_epoch(model, loader, criterion, optimizer, device):

    model.train()

    total_loss = 0.0

    for X_batch, y_batch in loader:

        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()

        logits = model(X_batch)

        loss = criterion(logits, y_batch)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def validate(model, loader, criterion, device):

    model.eval()

    total_loss = 0.0

    with torch.no_grad():

        for X_batch, y_batch in loader:

            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(X_batch)

            loss = criterion(logits, y_batch)

            total_loss += loss.item()

    return total_loss / len(loader)

def train_model(X_train, y_train, X_val, y_val):

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    train_dataset = ChurnDataset(X_train, y_train)
    val_dataset = ChurnDataset(X_val, y_val)

    train_loader = DataLoader(
        train_dataset,
        batch_size=1024,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1024,
        shuffle=False
    )

    model = ChurnANN(
        input_features=X_train.shape[1]
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )

    return model, train_loader, val_loader, criterion, optimizer, device