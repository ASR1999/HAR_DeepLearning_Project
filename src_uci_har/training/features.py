import argparse
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim

import src_uci_har.config as config
from src_uci_har.data_loader import get_feature_loaders
from src_uci_har.models import MLPClassifier


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    run_loss, correct, total = 0.0, 0, 0
    for X, y in loader:
        X, y = X.to(device).float(), y.to(device)
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        run_loss += loss.item()
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    return run_loss / len(loader), 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    run_loss, correct, total = 0.0, 0, 0
    for X, y in loader:
        X, y = X.to(device).float(), y.to(device)
        logits = model(X)
        loss = criterion(logits, y)
        run_loss += loss.item()
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    return run_loss / len(loader), 100.0 * correct / total


def main():
    parser = argparse.ArgumentParser(description="Train MLP on UCI/HAPT features or combined")
    parser.add_argument("--source", choices=["uci", "hapt", "combined"], default="combined")
    parser.add_argument("--epochs", type=int, default=config.N_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    args = parser.parse_args()

    if config.DISABLE_CUDNN:
        torch.backends.cudnn.enabled = False

    print(f"Using device: {config.DEVICE}")
    os.makedirs("./saved_models", exist_ok=True)

    train_loader, test_loader, n_classes = get_feature_loaders(args.source, args.batch_size)
    input_dim = next(iter(train_loader))[0].shape[1]

    model = MLPClassifier(input_dim=input_dim, n_classes=n_classes, dropout=config.DROPOUT).to(config.DEVICE)
    print(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best = 0.0
    start = time.time()
    for epoch in range(args.epochs):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, config.DEVICE)
        te_loss, te_acc = evaluate(model, test_loader, criterion, config.DEVICE)
        print(f"Epoch {epoch+1}/{args.epochs} | Train Loss {tr_loss:.4f} Acc {tr_acc:.2f}% | Val Loss {te_loss:.4f} Acc {te_acc:.2f}%")
        if te_acc > best:
            best = te_acc
            torch.save(model.state_dict(), f"./saved_models/best_mlp_{args.source}.pth")
            print(f"\u2713 Saved new best model (acc={best:.2f}%)")
    dur = (time.time() - start) / 60
    print(f"\nTraining finished in {dur:.2f} min. Best val acc: {best:.2f}%")


if __name__ == "__main__":
    main()


