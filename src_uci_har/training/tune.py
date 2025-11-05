import argparse
import csv
import os
import time
from itertools import product

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import src_uci_har.config as config
from src_uci_har.data_loader import get_data_loaders
from src_uci_har.models import CnnLstmModel


def train_one_epoch(model, loader, criterion, optimizer, device):
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
    parser = argparse.ArgumentParser(description="Hyperparameter tuning for CNN-LSTM (signals)")
    parser.add_argument("--epochs", type=int, default=8, help="epochs per trial")
    parser.add_argument("--results_dir", type=str, default="./results", help="where to write tuning results")
    parser.add_argument("--save_best", action="store_true", help="save best model to saved_models/best_model_tuned.pth")
    args = parser.parse_args()

    if config.DISABLE_CUDNN:
        torch.backends.cudnn.enabled = False
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    if config.DEVICE == "cuda" and torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.SEED)

    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs("./saved_models", exist_ok=True)

    # Search space (small and reasonable; adjust as needed)
    space = {
        "learning_rate": [1e-3, 5e-4, 1e-4],
        "batch_size": [32, 64, 128],
        "hidden_size": [64, 100, 128],
        "num_layers": [1, 2],
        "dropout": [0.3, 0.5],
    }

    keys = list(space.keys())
    combos = list(product(*[space[k] for k in keys]))

    # Fixed loaders per batch size? We'll recreate per trial to ensure correctness
    results_csv = os.path.join(args.results_dir, "tuning_signals_results.csv")
    write_header = not os.path.exists(results_csv)
    with open(results_csv, "a", newline="") as fcsv:
        writer = csv.writer(fcsv)
        if write_header:
            writer.writerow(["learning_rate", "batch_size", "hidden_size", "num_layers", "dropout", "val_acc", "val_loss", "epochs", "duration_min"])

        best_acc = -1.0
        best_cfg = None
        best_path = None

        for combo in combos:
            hp = {k: v for k, v in zip(keys, combo)}
            print(f"\nTrial: {hp}")
            train_loader, test_loader = get_data_loaders(hp["batch_size"])  # using test as val for simplicity

            model = CnnLstmModel(
                n_features=config.N_FEATURES,
                n_classes=config.N_CLASSES,
                hidden_size=hp["hidden_size"],
                num_layers=hp["num_layers"],
                dropout=hp["dropout"],
            ).to(config.DEVICE)
            if config.DISABLE_CUDNN:
                try:
                    model.lstm.flatten_parameters = lambda *args, **kwargs: None
                except Exception:
                    pass

            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=hp["learning_rate"])

            start = time.time()
            best_val_acc = -1.0
            for epoch in range(args.epochs):
                tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, config.DEVICE)
                va_loss, va_acc = evaluate(model, test_loader, criterion, config.DEVICE)
                print(f"  Epoch {epoch+1}/{args.epochs} | Train {tr_loss:.4f}/{tr_acc:.2f}% | Val {va_loss:.4f}/{va_acc:.2f}%")
                if va_acc > best_val_acc:
                    best_val_acc = va_acc
            dur = (time.time() - start) / 60

            writer.writerow([hp["learning_rate"], hp["batch_size"], hp["hidden_size"], hp["num_layers"], hp["dropout"], f"{best_val_acc:.4f}", f"{va_loss:.4f}", args.epochs, f"{dur:.2f}"])

            if best_val_acc > best_acc:
                best_acc = best_val_acc
                best_cfg = hp.copy()
                if args.save_best:
                    best_path = "./saved_models/best_model_tuned.pth"
                    torch.save(model.state_dict(), best_path)

        # Write summary TXT
        txt_path = os.path.join(args.results_dir, "tuning_signals_summary.txt")
        with open(txt_path, "w") as ftxt:
            ftxt.write("Best validation accuracy: %.4f\n" % best_acc)
            ftxt.write("Best config:\n")
            for k in keys:
                ftxt.write(f"  {k}: {best_cfg[k]}\n")
            if best_path:
                ftxt.write(f"Saved best model to: {best_path}\n")
        print(f"\nTuning complete. Best val acc={best_acc:.2f}%. Results: {results_csv}. Summary: {txt_path}")


if __name__ == "__main__":
    main()


