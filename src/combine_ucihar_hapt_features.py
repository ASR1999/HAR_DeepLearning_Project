# src/combine_ucihar_hapt_features.py
"""
Combine UCI HAR and HAPT feature datasets (561-dim) into one dataset filtered to the 6 core activities.
Outputs: data/combined_features/{X_train.npy, y_train.npy, X_test.npy, y_test.npy}
"""
import os
import numpy as np
from src.data_loader_features import combine_ucihar_hapt_features


OUT_DIR = "./data/combined_features"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    X_train, y_train, X_test, y_test = combine_ucihar_hapt_features()
    np.save(os.path.join(OUT_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(OUT_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(OUT_DIR, "X_test.npy"), X_test)
    np.save(os.path.join(OUT_DIR, "y_test.npy"), y_test)
    print(f"Saved combined features to {OUT_DIR}")
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")


if __name__ == "__main__":
    main()


