from collections import Counter
from typing import List

import matplotlib.pyplot as plt


def plot_grouped_counts(labels: List[str], title: str, save_path: str):
    counts = Counter(labels)
    keys = list(counts.keys())
    vals = [counts[k] for k in keys]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(keys, vals, color="#4C78A8")
    plt.title(title)
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    for rect in bars:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2.0, height, f"{int(height)}", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_true_vs_pred(true_labels: List[str], pred_labels: List[str], title: str, save_path: str):
    true_counts = Counter(true_labels)
    pred_counts = Counter(pred_labels)
    keys = sorted(set(true_counts.keys()) | set(pred_counts.keys()))
    true_vals = [true_counts.get(k, 0) for k in keys]
    pred_vals = [pred_counts.get(k, 0) for k in keys]

    x = range(len(keys))
    width = 0.38

    plt.figure(figsize=(10, 6))
    plt.bar([i - width/2 for i in x], true_vals, width=width, label="True", color="#72B7B2")
    plt.bar([i + width/2 for i in x], pred_vals, width=width, label="Pred", color="#4C78A8")
    plt.title(title)
    plt.ylabel("Count")
    plt.xticks(list(x), keys, rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


