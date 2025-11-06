import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import src_uci_har.config as config


class HARDataset(Dataset):
    def __init__(self, signal_paths, labels_path):
        self.signals = []
        for path in signal_paths:
            signal = pd.read_csv(path, sep=r"\s+", header=None, engine="python").values
            self.signals.append(signal)

        self.signals = np.stack(self.signals, axis=1)
        self.labels = pd.read_csv(labels_path, sep=r"\s+", header=None, engine="python").values.flatten() - 1

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        data = self.signals[idx]
        label = self.labels[idx]
        return torch.tensor(data, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


def get_data_loaders(batch_size):
    train_dataset = HARDataset(config.TRAIN_SIGNALS_PATH, config.TRAIN_LABELS_PATH)
    test_dataset = HARDataset(config.TEST_SIGNALS_PATH, config.TEST_LABELS_PATH)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


