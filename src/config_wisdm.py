# src/config_wisdm.py
import torch

# WISDM-specific settings
N_FEATURES = 3            # accelerometer only
SEQUENCE_LENGTH = 200     # 10s @ 20 Hz
N_CLASSES = 5             # WALKING, JOGGING, STAIRS, SITTING, STANDING

# Training hyperparameters (can be overridden via main args if needed)
BATCH_SIZE = 64
LEARNING_RATE = 0.001
N_EPOCHS = 20
DROPOUT = 0.5

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

