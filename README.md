# Human Activity Recognition (HAR) with PyTorch

This repository provides Human Activity Recognition models implemented in PyTorch for the UCI HAR and HAPT datasets. It includes a CNN-LSTM model for raw inertial signals and an MLP model for 561-dimensional engineered features. The modular package `src_uci_har/` is recommended for UCI/HAPT-only workflows.

Course: Deep Learning (CS-XXX)
Degree: M.Tech Data Science and Data Engineering, IIT Jodhpur
Group Members: Aditya Singh Rathor,Bratati, Jeyadev L, Sharone

---

## Repository Structure

```
HAR_DeepLearning_Project/
├── data/
│   ├── download_dataset.py          # Downloads and standardizes UCI HAR and HAPT
│   ├── uci_har/                     # UCI HAR (created by downloader)
│   └── hapt/                        # HAPT (created by downloader)
├── notebooks/
│   └── 01_data_exploration.ipynb
├── saved_models/
│   ├── best_model.pth               # CNN-LSTM on UCI signals
│   ├── best_mlp_{uci|hapt|combined}.pth
│   ├── training_history.png         # Loss/accuracy curves (signals)
│   ├── confusion_matrix.png         # Confusion matrix (signals)
│   ├── inference_grouped.png        # Predictions grouped by activity
│   └── inference_true_vs_pred.png   # True vs Predicted counts per activity
├── src/                             # Original reference implementation
├── src_uci_har/
│   ├── config.py
│   ├── data_loader/{signals.py,features.py}
│   ├── models/{cnn_lstm.py,mlp.py}
│   ├── training/{signals.py,features.py}
│   ├── evaluation/signals.py
│   └── inference/{__main__.py,random_grouped.py}
├── training_results1.txt
├── training_results2_combined.txt
├── requirements.txt
└── README.md
```

---

## Setup

Prerequisites: Python 3.8+, pip, optional CUDA GPU.

Install dependencies
```bash
pip install -r requirements.txt
```

Download datasets (UCI HAR and HAPT)
```bash
python data/download_dataset.py
```
The downloader creates standardized folders:
- `data/uci_har` (UCI HAR)
- `data/hapt` (HAPT)
If legacy folders exist, the code falls back to them:
- `data/UCI HAR Dataset/UCI HAR Dataset`, `data/HAPT`

---

## How to Run (src_uci_har)

Train CNN-LSTM on UCI signals
```bash
python -m src_uci_har.train
```

Evaluate CNN-LSTM
```bash
python -m src_uci_har.evaluate
```

Train MLP on features
```bash
python -m src_uci_har.train_features --source uci
python -m src_uci_har.train_features --source hapt
python -m src_uci_har.train_features --source combined
```

Inference with plots
```bash
# Signals (CNN-LSTM)
python -m src_uci_har.inference --mode signals --num_samples 100 \
  --out ./results/inference_grouped.png \
  --out_compare ./results/inference_true_vs_pred.png

# Features (MLP)
python -m src_uci_har.inference --mode features --source combined --num_samples 100 \
  --out ./results/inference_grouped.png \
  --out_compare ./results/inference_true_vs_pred.png
```

Environment controls
```bash
# Force CPU
HAR_FORCE_CPU=1 python -m src_uci_har.train
# Disable cuDNN for RNNs
HAR_DISABLE_CUDNN=1 python -m src_uci_har.train
# Explicit device
HAR_DEVICE=cuda python -m src_uci_har.train
```

Hyperparameter tuning (signals)
```bash
python -m src_uci_har.training.tune --epochs 8 --results_dir ./results --save_best
```
Writes `tuning_signals_results.csv` and `tuning_signals_summary.txt` under `./results`, and optionally saves `saved_models/best_model_tuned.pth`.

---

## Datasets

- UCI HAR: 30 subjects, 6 activities, 50 Hz; inertial windows of length 128.
- HAPT: 561-dim features; filtered to 6 shared activities when combined with UCI.

Standardized paths used by the code:
- UCI HAR: `data/uci_har/`
- HAPT: `data/hapt/`

---

## Models

CNN-LSTM for signals (UCI)
```
Input: (batch, 6, 128)
Conv1D(6→64, k=3) → ReLU → MaxPool(2)
Conv1D(64→128, k=3) → ReLU → MaxPool(2)
Permute → LSTM(2 layers, hidden=100, dropout=0.5)
FC(100→6)
```

MLP for features (UCI/HAPT)
```
Input: (batch, 561)
Linear(561→512) → ReLU → Dropout
Linear(512→256) → ReLU → Dropout
Linear(256→6)
```

Hyperparameters are defined in `src_uci_har/config.py` (batch size, learning rate, epochs, etc.).

---

## Data Exploration (notebooks/01_data_exploration.ipynb)

The notebook includes:
- Class distribution and basic statistics for UCI HAR
- Example inertial windows per activity (accelerometer and gyroscope)
- Correlation heatmaps
- Overview of 561 features and normalization ranges

---

## Results and Artifacts

Images (saved_models/)
- `training_history.png`: Loss and accuracy over epochs for signals training.
- `confusion_matrix.png`: Test-set confusion matrix for signals.
- `inference_grouped.png`: Predicted activity counts from random test samples.
- `inference_true_vs_pred.png`: True vs Predicted counts per activity on the same subset.

Text reports
- `training_results1.txt`: Example training log for signals (epoch-wise loss/accuracy).
- `training_results2_combined.txt`: Example training log for features on combined dataset.

Model checkpoints
- `best_model.pth`, `best_mlp_{uci|hapt|combined}.pth` under `saved_models/`.

---

## Expected Performance

- Test accuracy: typically 90–93% on UCI HAR with default settings.
- Training time: ~3–5 minutes (GPU) or 15–20 minutes (CPU).

---

## Troubleshooting

No module named `src_uci_har`
- Run commands with `-m` from the project root.

FileNotFoundError for data
- Run: `python data/download_dataset.py` and verify `data/uci_har`, `data/hapt`.

CUDA out of memory
- Lower `BATCH_SIZE` in `src_uci_har/config.py`.

---

## License

This project is for educational purposes as part of the Deep Learning course at IIT Jodhpur.

---

## Acknowledgments

- UCI Machine Learning Repository for the HAR dataset
- PyTorch team for the deep learning framework
- Course instructors and TAs for guidance

---

