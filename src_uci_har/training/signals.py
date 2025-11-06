import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import src_uci_har.config as config
from src_uci_har.data_loader import get_data_loaders
from src_uci_har.models import CnnLstmModel


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    run_loss, correct, total = 0.0, 0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        run_loss += loss.item()
        preds = logits.argmax(dim=1)
        total += y.size(0)
        correct += (preds == y).sum().item()
    return run_loss / len(loader), 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    run_loss, correct, total = 0.0, 0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        logits = model(X)
        loss = criterion(logits, y)
        run_loss += loss.item()
        preds = logits.argmax(dim=1)
        total += y.size(0)
        correct += (preds == y).sum().item()
    return run_loss / len(loader), 100.0 * correct / total


def main():
    if config.DISABLE_CUDNN:
        torch.backends.cudnn.enabled = False
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    if config.DEVICE == "cuda" and torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.SEED)

    print(f"Using device: {config.DEVICE}")
    os.makedirs("./saved_models", exist_ok=True)

    train_loader, test_loader = get_data_loaders(config.BATCH_SIZE)
    print(f"Train samples: {len(train_loader.dataset)} | Test samples: {len(test_loader.dataset)}")

    model = CnnLstmModel(
        n_features=config.N_FEATURES,
        n_classes=config.N_CLASSES,
        hidden_size=config.HIDDEN_SIZE,
        num_layers=config.NUM_LAYERS,
        dropout=config.DROPOUT,
    )
    if config.DISABLE_CUDNN:
        try:
            model.lstm.flatten_parameters = lambda *args, **kwargs: None
        except Exception:
            pass
    model = model.to(config.DEVICE)
    print(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    best = 0.0
    start = time.time()
    for epoch in range(config.N_EPOCHS):
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, config.DEVICE)
        te_loss, te_acc = evaluate(model, test_loader, criterion, config.DEVICE)
        print(f"Epoch {epoch+1}/{config.N_EPOCHS} | Train Loss {tr_loss:.4f} Acc {tr_acc:.2f}% | Val Loss {te_loss:.4f} Acc {te_acc:.2f}%")
        if te_acc > best:
            best = te_acc
            torch.save(model.state_dict(), "./saved_models/best_model.pth")
            print(f"\u2713 Saved new best model (acc={best:.2f}%)")
    dur = (time.time() - start) / 60
    print(f"\nTraining finished in {dur:.2f} min. Best val acc: {best:.2f}%")


if __name__ == "__main__":
    main()


