import os
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

import src_uci_har.config as config
from src_uci_har.data_loader import get_data_loaders
from src_uci_har.models import CnnLstmModel


def plot_confusion_matrix(cm, class_names, save_path):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix - HAR Model', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    os.makedirs("./saved_models", exist_ok=True)
    _, test_loader = get_data_loaders(config.BATCH_SIZE)

    model = CnnLstmModel(
        n_features=config.N_FEATURES,
        n_classes=config.N_CLASSES,
        hidden_size=config.HIDDEN_SIZE,
        num_layers=config.NUM_LAYERS,
        dropout=0.0,
    ).to(config.DEVICE)
    model.load_state_dict(torch.load("./saved_models/best_model.pth", map_location=config.DEVICE))
    model.eval()

    all_labels, all_preds = [], []
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(config.DEVICE), y.to(config.DEVICE)
            logits = model(X)
            preds = logits.argmax(dim=1)
            all_labels.extend(y.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    accuracy = 100 * np.mean(np.array(all_labels) == np.array(all_preds))
    print(f"Overall Test Accuracy: {accuracy:.2f}%")
    print(classification_report(all_labels, all_preds, target_names=config.ACTIVITY_NAMES, digits=4))
    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(cm, config.ACTIVITY_NAMES, "./saved_models/confusion_matrix.png")


if __name__ == "__main__":
    main()


