import argparse
import os
import random
from typing import List

import numpy as np
import torch

import src_uci_har.config as cfg
from src_uci_har.models import CnnLstmModel, MLPClassifier
from src_uci_har.data_loader import get_data_loaders, get_feature_loaders
from src_uci_har.utils.plots import plot_grouped_counts, plot_true_vs_pred


def predict_signals_random_samples(num_samples: int, save_path: str):
    _, test_loader = get_data_loaders(batch_size=1)
    model = CnnLstmModel(
        n_features=cfg.N_FEATURES,
        n_classes=cfg.N_CLASSES,
        hidden_size=cfg.HIDDEN_SIZE,
        num_layers=cfg.NUM_LAYERS,
        dropout=0.0,
    ).to(cfg.DEVICE)
    model.load_state_dict(torch.load("./saved_models/best_model.pth", map_location=cfg.DEVICE))
    model.eval()

    test_items: List[tuple[torch.Tensor, int]] = []
    for X, y in test_loader:
        test_items.append((X.squeeze(0), int(y.item())))

    chosen = random.sample(test_items, k=min(num_samples, len(test_items)))
    pred_names: List[str] = []
    true_names: List[str] = []
    with torch.no_grad():
        for X, y in chosen:
            logits = model(X.unsqueeze(0).to(cfg.DEVICE))
            pred = int(logits.argmax(dim=1).cpu().numpy()[0])
            pred_names.append(cfg.ACTIVITY_NAMES[pred])
            true_names.append(cfg.ACTIVITY_NAMES[y])

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plot_grouped_counts(pred_names, f"Signals predictions (N={len(pred_names)})", save_path)
    return pred_names, true_names


def predict_features_random_samples(source: str, num_samples: int, save_path: str):
    _, test_loader, n_classes = get_feature_loaders(source, batch_size=1)
    input_dim = next(iter(test_loader))[0].shape[1]
    model_path = f"./saved_models/best_mlp_{source}.pth"
    model = MLPClassifier(input_dim=input_dim, n_classes=n_classes, dropout=0.0).to(cfg.DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=cfg.DEVICE))
    model.eval()

    test_items: List[tuple[torch.Tensor, int]] = []
    for X, y in test_loader:
        test_items.append((X.squeeze(0), int(y.item())))

    chosen = random.sample(test_items, k=min(num_samples, len(test_items)))
    pred_names: List[str] = []
    true_names: List[str] = []
    with torch.no_grad():
        for X, y in chosen:
            logits = model(X.unsqueeze(0).to(cfg.DEVICE).float())
            pred = int(logits.argmax(dim=1).cpu().numpy()[0])
            pred_names.append(cfg.ACTIVITY_NAMES[pred])
            true_names.append(cfg.ACTIVITY_NAMES[y])

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plot_grouped_counts(pred_names, f"Features predictions (N={len(pred_names)})", save_path)
    return pred_names, true_names


def main():
    parser = argparse.ArgumentParser(description="Inference with random sampling and grouped plots")
    parser.add_argument("--mode", choices=["signals", "features"], default="signals")
    parser.add_argument("--source", choices=["uci", "hapt", "combined"], default="combined",
                        help="Only used in features mode")
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--out", type=str, default="./saved_models/inference_grouped.png")
    parser.add_argument("--out_compare", type=str, default="./saved_models/inference_true_vs_pred.png")
    args = parser.parse_args()

    random.seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    torch.manual_seed(cfg.SEED)

    if args.mode == "signals":
        preds, trues = predict_signals_random_samples(args.num_samples, args.out)
    else:
        preds, trues = predict_features_random_samples(args.source, args.num_samples, args.out)

    plot_true_vs_pred(trues, preds, f"True vs Pred (N={len(preds)})", args.out_compare)
    print(f"Saved grouped prediction plot to {args.out}")
    print(f"Saved true vs pred comparison plot to {args.out_compare}")


if __name__ == "__main__":
    main()


