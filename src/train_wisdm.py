# src/train_wisdm.py
import time
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.data_loader_wisdm import get_wisdm_loaders
from src.model import CnnLstmModel
import src.config_wisdm as cfg
import src.config as base_cfg


def train(model, loader, criterion, optimizer, device):
    model.train()
    total = 0
    correct = 0
    run_loss = 0.0
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
    total = 0
    correct = 0
    run_loss = 0.0
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
    if base_cfg.DISABLE_CUDNN:
        torch.backends.cudnn.enabled = False
    torch.manual_seed(base_cfg.SEED)
    np.random.seed(base_cfg.SEED)
    if cfg.DEVICE == "cuda" and torch.cuda.is_available():
        torch.cuda.manual_seed_all(base_cfg.SEED)

    print(f"Using device: {cfg.DEVICE}")
    os.makedirs("./saved_models", exist_ok=True)
    train_loader, test_loader, n_classes = get_wisdm_loaders(batch_size=cfg.BATCH_SIZE)
    print(f"WISDM windows -> train: {len(train_loader.dataset)}, test: {len(test_loader.dataset)}")

    model = CnnLstmModel(
        n_features=cfg.N_FEATURES,
        n_classes=n_classes,
        hidden_size=base_cfg.HIDDEN_SIZE,
        num_layers=base_cfg.NUM_LAYERS,
        dropout=base_cfg.DROPOUT,
    )
    if base_cfg.DISABLE_CUDNN:
        try:
            model.lstm.flatten_parameters = lambda *args, **kwargs: None
        except Exception:
            pass
    model = model.to(cfg.DEVICE)
    print(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)
    best = 0.0
    start = time.time()
    for epoch in range(cfg.N_EPOCHS):
        tr_loss, tr_acc = train(model, train_loader, criterion, optimizer, cfg.DEVICE)
        te_loss, te_acc = evaluate(model, test_loader, criterion, cfg.DEVICE)
        print(f"Epoch {epoch+1}/{cfg.N_EPOCHS} | Train Loss {tr_loss:.4f} Acc {tr_acc:.2f}% | Val Loss {te_loss:.4f} Acc {te_acc:.2f}%")
        if te_acc > best:
            best = te_acc
            torch.save(model.state_dict(), "./saved_models/best_wisdm_cnnlstm.pth")
            print(f"✓ Saved new best model (acc={best:.2f}%)")
    dur = (time.time() - start) / 60
    print(f"\nTraining finished in {dur:.2f} min. Best val acc: {best:.2f}%")


if __name__ == "__main__":
    main()


