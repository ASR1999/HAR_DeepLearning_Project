# src/data_loader.py
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import src.config as config

class HARDataset(Dataset):
    def __init__(self, signal_paths, labels_path):
        self.signals = []
        for path in signal_paths:
            # Load as DataFrame, split by whitespace, and convert to numpy
            signal = pd.read_csv(path, delim_whitespace=True, header=None).values
            self.signals.append(signal)
        
        # Stack signals to get shape (n_samples, n_channels, sequence_length)
        # We stack along a new axis (axis 1)
        self.signals = np.stack(self.signals, axis=1) 
        
        # Load labels
        # Labels are 1-indexed, so subtract 1 to make them 0-indexed
        self.labels = pd.read_csv(labels_path, header=None).values.flatten() - 1

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Get data and label
        data = self.signals[idx]
        label = self.labels[idx]
        
        # Convert to PyTorch tensors
        # Data shape: [N_FEATURES, SEQUENCE_LENGTH]
        # Label shape: [1]
        return torch.tensor(data, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

def get_data_loaders(batch_size):
    train_dataset = HARDataset(config.TRAIN_SIGNALS_PATH, config.TRAIN_LABELS_PATH)
    test_dataset = HARDataset(config.TEST_SIGNALS_PATH, config.TEST_LABELS_PATH)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader