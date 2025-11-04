# src/data_loader_features.py
import os
from typing import Tuple, Literal

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


UCI_DIR = "./data/UCI HAR Dataset/UCI HAR Dataset/"
HAPT_DIR = "./data/HAPT/"


def _read_txt_matrix(path: str) -> np.ndarray:
    return pd.read_csv(path, sep=r"\s+", header=None, engine="python").values


def load_uci_features() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_train = _read_txt_matrix(os.path.join(UCI_DIR, "train", "X_train.txt"))
    y_train = _read_txt_matrix(os.path.join(UCI_DIR, "train", "y_train.txt")).flatten()
    X_test = _read_txt_matrix(os.path.join(UCI_DIR, "test", "X_test.txt"))
    y_test = _read_txt_matrix(os.path.join(UCI_DIR, "test", "y_test.txt")).flatten()

    # Keep only the 6 core activities to align with UCI HAR classes
    train_mask = y_train <= 6
    test_mask = y_test <= 6
    X_train, y_train = X_train[train_mask], y_train[train_mask] - 1
    X_test, y_test = X_test[test_mask], y_test[test_mask] - 1
    return X_train, y_train, X_test, y_test


def load_hapt_features() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_train = _read_txt_matrix(os.path.join(HAPT_DIR, "Train", "X_train.txt"))
    y_train = _read_txt_matrix(os.path.join(HAPT_DIR, "Train", "y_train.txt")).flatten()
    X_test = _read_txt_matrix(os.path.join(HAPT_DIR, "Test", "X_test.txt"))
    y_test = _read_txt_matrix(os.path.join(HAPT_DIR, "Test", "y_test.txt")).flatten()

    # Keep only the 6 core activities to align with UCI HAR classes
    train_mask = y_train <= 6
    test_mask = y_test <= 6
    X_train, y_train = X_train[train_mask], y_train[train_mask] - 1
    X_test, y_test = X_test[test_mask], y_test[test_mask] - 1
    return X_train, y_train, X_test, y_test


def combine_ucihar_hapt_features() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    Xu_tr, yu_tr, Xu_te, yu_te = load_uci_features()
    Xh_tr, yh_tr, Xh_te, yh_te = load_hapt_features()

    X_train = np.vstack([Xu_tr, Xh_tr])
    y_train = np.concatenate([yu_tr, yh_tr])
    X_test = np.vstack([Xu_te, Xh_te])
    y_test = np.concatenate([yu_te, yh_te])
    return X_train, y_train, X_test, y_test


class FeaturesDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, mean: np.ndarray | None = None, std: np.ndarray | None = None):
        assert X.ndim == 2, "X must be 2D [n_samples, n_features]"
        self.X = X.astype(np.float32, copy=False)
        self.y = y.astype(np.int64, copy=False)
        # Standardize (ensure float32 everywhere)
        if mean is None or std is None:
            mean = self.X.mean(axis=0).astype(np.float32, copy=False)
            std = (self.X.std(axis=0) + 1e-8).astype(np.float32, copy=False)
        self.mean = mean.astype(np.float32, copy=False)
        self.std = std.astype(np.float32, copy=False)
        self.X = ((self.X - self.mean) / self.std).astype(np.float32, copy=False)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.tensor(self.y[idx], dtype=torch.long)


def get_feature_loaders(source: Literal["uci", "hapt", "combined"], batch_size: int) -> Tuple[DataLoader, DataLoader, int]:
    if source == "uci":
        X_train, y_train, X_test, y_test = load_uci_features()
    elif source == "hapt":
        X_train, y_train, X_test, y_test = load_hapt_features()
    elif source == "combined":
        X_train, y_train, X_test, y_test = combine_ucihar_hapt_features()
    else:
        raise ValueError("source must be one of: 'uci', 'hapt', 'combined'")

    # Fit normalization on train (float32)
    X_train = X_train.astype(np.float32, copy=False)
    X_test = X_test.astype(np.float32, copy=False)
    mean = X_train.mean(axis=0).astype(np.float32, copy=False)
    std = (X_train.std(axis=0) + 1e-8).astype(np.float32, copy=False)
    train_ds = FeaturesDataset(X_train, y_train, mean, std)
    test_ds = FeaturesDataset(X_test, y_test, mean, std)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    n_classes = int(max(y_train.max(), y_test.max())) + 1 if len(y_train) > 0 and len(y_test) > 0 else 6
    return train_loader, test_loader, n_classes


