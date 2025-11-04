# src/data_loader_wisdm.py
import os
import glob
from collections import Counter, defaultdict
from typing import Tuple, List

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split


WISDM_DIR = "./data/WISDM/wisdm-dataset/"
RAW_DIR = os.path.join(WISDM_DIR, "raw", "phone", "accel")
ACTIVITY_KEY_PATH = os.path.join(WISDM_DIR, "activity_key.txt")

# We will train on 5 classes available in raw phone accelerometer
# A: walking, B: jogging, C: stairs, D: sitting, E: standing
ALLOWED_CODES = ["A", "B", "C", "D", "E"]
CLASS_ORDER = ["WALKING", "JOGGING", "STAIRS", "SITTING", "STANDING"]


def load_activity_key() -> dict:
    mapping: dict[str, str] = {}
    with open(ACTIVITY_KEY_PATH, "r") as f:
        for line in f:
            line = line.strip()
            if not line or "=" not in line:
                continue
            name, code = line.split("=")
            name = name.strip()
            code = code.strip()
            mapping[code] = name
    return mapping


def parse_wisdm_file(path: str) -> tuple[np.ndarray, list[str], int]:
    # Returns arrays of shape [N, 3], labels per-sample (letters), and subject id
    xs: List[float] = []
    ys: List[float] = []
    zs: List[float] = []
    labels: List[str] = []
    subject_id = None
    with open(path, "r") as f:
        for raw in f:
            raw = raw.strip().rstrip(";")
            if not raw:
                continue
            parts = raw.split(",")
            if len(parts) < 6:
                continue
            sid, code, ts, x, y, z = parts[:6]
            if subject_id is None:
                try:
                    subject_id = int(sid)
                except Exception:
                    subject_id = -1
            code = code.strip()
            if code not in ALLOWED_CODES:
                continue
            try:
                xs.append(float(x))
                ys.append(float(y))
                zs.append(float(z))
                labels.append(code)
            except Exception:
                continue
    data = np.stack([xs, ys, zs], axis=1) if xs else np.zeros((0, 3), dtype=np.float32)
    return data.astype(np.float32), labels, (subject_id or -1)


def create_windows(data: np.ndarray, labels: List[str], window_size: int = 200, stride: int = 100, min_majority: float = 0.8):
    # data: [N, 3]
    # returns X: [M, 3, window_size], y: [M]
    X_list = []
    y_list = []
    n = data.shape[0]
    for start in range(0, max(n - window_size + 1, 0), stride):
        end = start + window_size
        window_labels = labels[start:end]
        if len(window_labels) < window_size:
            break
        counts = Counter(window_labels)
        major_code, major_count = counts.most_common(1)[0]
        if major_count / window_size < min_majority:
            continue
        xw = data[start:end].T  # (3, window)
        X_list.append(xw)
        y_list.append(ALLOWED_CODES.index(major_code))
    if not X_list:
        return np.zeros((0, 3, window_size), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    X = np.stack(X_list, axis=0).astype(np.float32)
    y = np.array(y_list, dtype=np.int64)
    return X, y


class WISDMDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X
        self.y = y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(int(self.y[idx]), dtype=torch.long)


def get_wisdm_loaders(batch_size: int, window_size: int = 200, stride: int = 100, subject_split: float = 0.2) -> Tuple[DataLoader, DataLoader, int]:
    files = sorted(glob.glob(os.path.join(RAW_DIR, "data_*_accel_phone.txt")))
    subjects_to_samples: dict[int, list[tuple[np.ndarray, np.ndarray]]] = defaultdict(list)

    for fp in files:
        data, labels, sid = parse_wisdm_file(fp)
        if data.shape[0] == 0:
            continue
        X, y = create_windows(data, labels, window_size=window_size, stride=stride)
        if X.shape[0] == 0:
            continue
        subjects_to_samples[sid].append((X, y))

    # Concatenate per subject
    subject_ids = sorted(subjects_to_samples.keys())
    if not subject_ids:
        raise RuntimeError("No WISDM samples found. Ensure files exist under raw/phone/accel/")

    split_idx = int(len(subject_ids) * (1.0 - subject_split))
    train_subjects = set(subject_ids[:split_idx])
    test_subjects = set(subject_ids[split_idx:])

    def concat(subj_set: set[int]) -> tuple[np.ndarray, np.ndarray]:
        Xs, ys = [], []
        for sid in subj_set:
            for X, y in subjects_to_samples[sid]:
                Xs.append(X)
                ys.append(y)
        return np.concatenate(Xs, axis=0), np.concatenate(ys, axis=0)

    X_train, y_train = concat(train_subjects)
    X_test, y_test = concat(test_subjects)

    train_ds = WISDMDataset(X_train, y_train)
    test_ds = WISDMDataset(X_test, y_test)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    n_classes = len(ALLOWED_CODES)
    return train_loader, test_loader, n_classes


