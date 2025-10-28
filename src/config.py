# src/config.py
import torch

# Data paths
DATA_DIR = "./data/UCI HAR Dataset/"
TRAIN_SIGNALS_PATH = [
    DATA_DIR + "train/Inertial Signals/body_acc_x_train.txt",
    DATA_DIR + "train/Inertial Signals/body_acc_y_train.txt",
    DATA_DIR + "train/Inertial Signals/body_acc_z_train.txt",
    DATA_DIR + "train/Inertial Signals/body_gyro_x_train.txt",
    DATA_DIR + "train/Inertial Signals/body_gyro_y_train.txt",
    DATA_DIR + "train/Inertial Signals/body_gyro_z_train.txt",
]
TRAIN_LABELS_PATH = DATA_DIR + "train/y_train.txt"

TEST_SIGNALS_PATH = [
    DATA_DIR + "test/Inertial Signals/body_acc_x_test.txt",
    DATA_DIR + "test/Inertial Signals/body_acc_y_test.txt",
    DATA_DIR + "test/Inertial Signals/body_acc_z_test.txt",
    DATA_DIR + "test/Inertial Signals/body_gyro_x_test.txt",
    DATA_DIR + "test/Inertial Signals/body_gyro_y_test.txt",
    DATA_DIR + "test/Inertial Signals/body_gyro_z_test.txt",
]
TEST_LABELS_PATH = DATA_DIR + "test/y_test.txt"

# Model Hyperparameters
N_CLASSES = 6           # 6 activities
SEQUENCE_LENGTH = 128   # 128 readings per window
N_FEATURES = 6          # 6 signals (acc_x, y, z; gyro_x, y, z)
HIDDEN_SIZE = 100
NUM_LAYERS = 2
DROPOUT = 0.5

# Training Hyperparameters
BATCH_SIZE = 64
LEARNING_RATE = 0.001
N_EPOCHS = 25

# Compute device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"