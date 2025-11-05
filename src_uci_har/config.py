import torch
import os

# Data paths (standardized, with legacy fallback for existing projects)
_UCI_STD = "./data/uci_har"
_UCI_LEGACY = "./data/UCI HAR Dataset/UCI HAR Dataset"
_HAPT_STD = "./data/hapt"
_HAPT_LEGACY = "./data/HAPT"

UCI_DIR = _UCI_STD if os.path.isdir(_UCI_STD) else _UCI_LEGACY
HAPT_DIR = _HAPT_STD if os.path.isdir(_HAPT_STD) else _HAPT_LEGACY

# Signals (from UCI inertial)
DATA_DIR = UCI_DIR
TRAIN_SIGNALS_PATH = [
    DATA_DIR + "/train/Inertial Signals/body_acc_x_train.txt",
    DATA_DIR + "/train/Inertial Signals/body_acc_y_train.txt",
    DATA_DIR + "/train/Inertial Signals/body_acc_z_train.txt",
    DATA_DIR + "/train/Inertial Signals/body_gyro_x_train.txt",
    DATA_DIR + "/train/Inertial Signals/body_gyro_y_train.txt",
    DATA_DIR + "/train/Inertial Signals/body_gyro_z_train.txt",
]
TRAIN_LABELS_PATH = DATA_DIR + "/train/y_train.txt"

TEST_SIGNALS_PATH = [
    DATA_DIR + "/test/Inertial Signals/body_acc_x_test.txt",
    DATA_DIR + "/test/Inertial Signals/body_acc_y_test.txt",
    DATA_DIR + "/test/Inertial Signals/body_acc_z_test.txt",
    DATA_DIR + "/test/Inertial Signals/body_gyro_x_test.txt",
    DATA_DIR + "/test/Inertial Signals/body_gyro_y_test.txt",
    DATA_DIR + "/test/Inertial Signals/body_gyro_z_test.txt",
]
TEST_LABELS_PATH = DATA_DIR + "/test/y_test.txt"

# Model Hyperparameters
N_CLASSES = 6
SEQUENCE_LENGTH = 128
N_FEATURES = 6
HIDDEN_SIZE = 100
NUM_LAYERS = 2
DROPOUT = 0.5

# Training Hyperparameters
BATCH_SIZE = 64
LEARNING_RATE = 0.001
N_EPOCHS = 25

# Reproducibility
SEED = 42

# Device and backend controls
FORCE_CPU = os.getenv("HAR_FORCE_CPU", "0") == "1"
DISABLE_CUDNN = os.getenv("HAR_DISABLE_CUDNN", "0") == "1"
_ENV_DEVICE = os.getenv("HAR_DEVICE")

if FORCE_CPU:
    DEVICE = "cpu"
elif _ENV_DEVICE in ("cpu", "cuda"):
    DEVICE = _ENV_DEVICE
else:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ACTIVITY_NAMES = [
    "WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS",
    "SITTING", "STANDING", "LAYING"
]


