# src/model.py
import torch
import torch.nn as nn
import src.config as config

class CnnLstmModel(nn.Module):
    def __init__(self, n_features, n_classes, hidden_size, num_layers, dropout):
        super(CnnLstmModel, self).__init__()
        
        self.n_features = n_features
        self.n_classes = n_classes
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # 1D Convolutional layers
        self.conv1 = nn.Conv1d(in_channels=n_features, out_channels=64, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        
        # LSTM layers
        # Input to LSTM: (batch_size, seq_len, input_size)
        # After convs/pools, seq_len is 128 / 2 / 2 = 32
        # After convs, input_size (features) is 128
        self.lstm = nn.LSTM(
            input_size=128, 
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,  # This is important!
            dropout=dropout
        )
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_size, n_classes)

    def forward(self, x):
        # Input x shape: (batch_size, n_features, seq_len) -> e.g., (64, 6, 128)
        
        # CNN part
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        # Shape: (64, 64, 64)
        
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        # Shape: (64, 128, 32)
        
        # Prepare for LSTM
        # LSTM expects (batch_size, seq_len, features)
        # We need to permute the dimensions
        x = x.permute(0, 2, 1)  # Shape: (64, 32, 128)
        
        # LSTM part
        # h0 and c0 are initialized to zero by default
        # out: (batch_size, seq_len, hidden_size) -> (64, 32, 100)
        # (hn, cn): last hidden/cell state
        out, (hn, cn) = self.lstm(x)
        
        # We only need the output from the last time step
        # out: (batch_size, hidden_size) -> (64, 100)
        out = out[:, -1, :]
        
        # Classifier part
        out = self.fc(out) # Shape: (64, n_classes) -> (64, 6)
        
        return out