# Deep Learning Project: Human Activity Recognition

This project implements a Deep Learning model (CNN-LSTM) in PyTorch to classify human activities from smartphone sensor data.

**Course:** Deep Learning (CS-XXX)
**Degree:** M.Tech Data Science and Data Engineering, IIT Jodhpur
**Group Members:** [Your Name(s)]

---

## 📋 Project Overview

This project implements two complementary HAR pipelines in PyTorch:
- A **hybrid CNN-LSTM (signals)** model on raw inertial windows `(6×128)`
- A lightweight **MLP (features)** model on 561-dim engineered features (UCI/HAPT)
Both achieve strong accuracy on the 6 core activities.

### Activities Recognized
1. WALKING
2. WALKING_UPSTAIRS
3. WALKING_DOWNSTAIRS
4. SITTING
5. STANDING
6. LAYING

### Key Features
- ✅ Implemented in **PyTorch**
- ✅ CNN layers for local feature extraction
- ✅ LSTM layers for temporal sequence modeling
- ✅ Complete training and evaluation pipeline
- ✅ Automated dataset download
- ✅ Comprehensive visualizations
- ✅ Saved model checkpoints

---

## 📁 Project Structure

```
HAR_DeepLearning_Project/
│
├── data/
│   ├── download_dataset.py                # Download UCI HAR
│   └── UCI HAR Dataset/                   # Dataset folder (created after download)
│
├── notebooks/
│   └── 01_data_exploration.ipynb          # Data exploration and visualization
│
├── saved_models/
│   ├── best_model.pth                     # CNN-LSTM (signals) checkpoint
│   ├── best_mlp_{uci|hapt|combined}.pth   # MLP (features) checkpoints
│   ├── training_history.png               # Training/validation curves
│   └── confusion_matrix.png               # Confusion matrix visualization
│
├── src/
│   ├── __init__.py
│   ├── config.py                          # Core config + device/env flags
│   ├── data_loader.py                     # Signals loader (6×128 windows)
│   ├── model.py                           # CNN-LSTM
│   ├── train.py                           # Train CNN-LSTM (signals)
│   ├── evaluate.py                        # Evaluate CNN-LSTM (signals)
│   ├── inference.py                       # Inference: signals or features
│   ├── data_loader_features.py            # Load 561-dim UCI/HAPT features
│   ├── model_mlp.py                       # MLP classifier for features
│   ├── train_features.py                  # Train MLP on {uci|hapt|combined}
│   ├── combine_ucihar_hapt_features.py    # Build combined features dataset
│   ├── data_loader_wisdm.py               # WISDM raw accel (3×200 windows)
│   ├── config_wisdm.py                    # WISDM-specific config
│   └── train_wisdm.py                     # Train CNN-LSTM on WISDM
│
├── requirements.txt
└── README.md
```

---

## 🔧 Setup and Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- (Optional) CUDA-capable GPU for faster training

### Step 1: Clone or Download the Project

```bash
# If using git
git clone <your-repository-url>
cd HAR_DeepLearning_Project

# Or simply download and extract the ZIP file
```

### Step 2: Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt
```

**Required Packages:**
- `torch` (PyTorch deep learning framework)
- `pandas` (Data manipulation)
- `numpy` (Numerical operations)
- `scikit-learn` (Metrics and evaluation)
- `matplotlib` (Plotting and visualization)
- `seaborn` (Statistical visualizations)
- `requests` (Dataset download)

### Step 3: Download the Dataset

The UCI HAR Dataset is automatically downloaded using our script:

```bash
# Run from project root directory
python data/download_dataset.py
```

**What this does:**
- Downloads the dataset from UCI Machine Learning Repository
- Extracts it to `data/UCI HAR Dataset/`
- The dataset is ~60 MB

**Manual Download (Alternative):**
If the script fails, you can manually:
1. Download from: https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip
2. Extract the ZIP file
3. Move `UCI HAR Dataset` folder to `data/` directory

---

## 🚀 How to Run

### Training the Model

Train the model from scratch:

```bash
python -m src.train
```

**What happens:**
- Loads training and test data
- Initializes the CNN-LSTM model
- Trains for 25 epochs (configurable in `config.py`)
- Saves best model to `saved_models/best_model.pth`
- Generates training history plot: `saved_models/training_history.png`
- Prints training progress and metrics

**Expected output:**
```
Using device: cuda  # or cpu
Training samples: 7352
Test samples: 2947

Model Architecture:
...
Total parameters: 234,662
Trainable parameters: 234,662

Starting training...
Epoch 1/25 | Train Loss: 1.2345, Train Acc: 55.23% | Val Loss: 0.9876, Val Acc: 65.12%
✓ New best model saved with accuracy: 65.12%
...
```

**Training Time:**
- CPU: ~15-20 minutes
- GPU: ~3-5 minutes

### Evaluating the Model

Evaluate the trained model on test set:

```bash
python -m src.evaluate
```

**What happens:**
- Loads the saved model from `saved_models/best_model.pth`
- Runs inference on test set
- Prints classification report with precision, recall, F1-score
- Prints confusion matrix
- Generates confusion matrix visualization: `saved_models/confusion_matrix.png`

**Expected output:**
```
============================================================
HAR MODEL EVALUATION
============================================================

Loading test data...
Test samples: 2947

Loading saved model from './saved_models/best_model.pth'...
✓ Model loaded successfully

Running inference on test set...

============================================================
TEST RESULTS
============================================================

Overall Test Accuracy: 92.35%

Classification Report:
------------------------------------------------------------
                      precision    recall  f1-score   support
...
```

### Making Predictions (Inference)

The inference script supports two modes via `--mode`:
- `signals`: CNN-LSTM on raw inertial windows (6×128)
- `features`: MLP on 561-dim features (UCI/HAPT/combined)

Run on signals (CNN-LSTM):
```bash
python -m src.inference --mode signals --model_path ./saved_models/best_model.pth
```

Run on features (MLP):
```bash
python -m src.inference --mode features --source combined \
  --model_path ./saved_models/best_mlp_combined.pth
```

Programmatic usage:
```python
# Signals (CNN-LSTM)
import numpy as np
from src.inference import SignalPredictor
pred = SignalPredictor("./saved_models/best_model.pth")
x = np.random.randn(6, 128).astype("float32")
res = pred.predict(x)

# Features (MLP)
import numpy as np
from src.inference import FeaturePredictor
feat = np.random.randn(561).astype("float32")  # ensure same normalization as train
pred2 = FeaturePredictor(input_dim=561, n_classes=6,
                         model_path="./saved_models/best_mlp_combined.pth").predict(feat)
```

### Data Exploration

Explore the dataset using Jupyter notebook:

```bash
# Start Jupyter notebook
jupyter notebook notebooks/01_data_exploration.ipynb
```

**What you'll see:**
- Dataset statistics and distribution
- Signal visualizations for each activity
- Statistical analysis of sensor data
- Heatmaps showing activity patterns

---

## Using Additional Datasets (HAPT and WISDM)

### Can we combine datasets?
- **UCI HAR + HAPT (features)**: YES. Both provide 561-dim feature vectors; we can safely combine after filtering to the 6 shared activities.
- **WISDM with UCI/HAPT**: NOT recommended directly. WISDM raw phone accelerometer is 3-axis at 20 Hz, label sets differ (e.g., single "stairs" class), and lacks gyroscope. Use a separate pipeline.

### Combine UCI + HAPT features and train
```bash
# 1) Create combined feature dataset
python -m src.combine_ucihar_hapt_features

# 2) Train an MLP classifier on combined features
python -m src.train_features --source combined

# (Optional) Train on UCI-only or HAPT-only features
python -m src.train_features --source uci
python -m src.train_features --source hapt
```

### Train on WISDM (raw phone accelerometer)
```bash
# This trains CNN-LSTM on 3-channel accel with 10s windows (200 samples @20Hz)
python -m src.train_wisdm
```

Notes:
- WISDM classes used: WALKING, JOGGING, STAIRS, SITTING, STANDING (5 classes).
- HAPT provides 12 labels; we filter to the 6 that overlap with UCI HAR.
 - Inference for WISDM is not wired in `src.inference`; use the training script or adapt the signals flow.

---

## 📖 File Descriptions

### Core Files

**`src/config.py`**
- Contains all configuration parameters
- Hyperparameters: learning rate, batch size, hidden size, etc.
- File paths for dataset
- Device configuration (CPU/GPU)

**`src/data_loader.py`**
- Defines `HARDataset` class (PyTorch Dataset)
- Loads raw inertial signal files
- Preprocessing: stacking 6 sensor channels
- Creates train and test DataLoaders
- Input shape: `(batch_size, 6, 128)`

**`src/model.py`**
- Defines `CnnLstmModel` (PyTorch nn.Module)
- Architecture:
  - **Conv1D layers**: Local feature extraction (2 layers)
  - **MaxPooling**: Dimensionality reduction
  - **LSTM layers**: Temporal sequence modeling (2 layers)
  - **Fully Connected**: Classification layer
- Forward pass implementation

**`src/train.py`**
- Main training script
- Training loop with backpropagation
- Validation after each epoch
- Model checkpointing (saves best model)
- Plots training history
- Uses Adam optimizer and CrossEntropyLoss

**`src/evaluate.py`**
- Loads saved model
- Runs inference on test set
- Computes metrics: accuracy, precision, recall, F1-score
- Generates confusion matrix
- Identifies misclassifications

**`src/inference.py`**
- `SignalPredictor` for CNN-LSTM (signals) and `FeaturePredictor` for MLP (features)
- CLI supports `--mode {signals,features}` and `--source {uci,hapt,combined}`
- Returns predictions with confidence scores; includes demo

**`data/download_dataset.py`**
### New files for additional datasets

**`src/data_loader_features.py`**: Loads 561-dim feature datasets (UCI/HAPT) and supports combining.

**`src/model_mlp.py`**: Simple MLP classifier for feature inputs.

**`src/train_features.py`**: Train the MLP on UCI, HAPT, or combined features.

**`src/combine_ucihar_hapt_features.py`**: Writes combined feature arrays to `data/combined_features/`.

**`src/data_loader_wisdm.py`**: Parses WISDM raw phone accelerometer, windows into 200-sample segments with majority label.

**`src/config_wisdm.py`** and **`src/train_wisdm.py`**: Config and training script for WISDM (3-channel CNN-LSTM).

---

- Automated dataset download
- Downloads from UCI repository
- Extracts ZIP file
- Error handling for network issues

---

## 🧠 Model Architecture

### CNN-LSTM Hybrid Model

```
Input: (batch_size, 6, 128)
    ↓
[Conv1D: 6→64, kernel=3] → ReLU → MaxPool(2)
    ↓
[Conv1D: 64→128, kernel=3] → ReLU → MaxPool(2)
    ↓
Reshape: (batch_size, 32, 128)
    ↓
[LSTM: 2 layers, hidden_size=100, dropout=0.5]
    ↓
Take last time step: (batch_size, 100)
    ↓
[Fully Connected: 100→6]
    ↓
Output: (batch_size, 6)
```

**Why this architecture?**
- **1D CNN**: Extracts local patterns in time-series data (feature extraction)
- **MaxPooling**: Reduces sequence length, computational efficiency
- **LSTM**: Captures long-term temporal dependencies
- **Dropout**: Prevents overfitting
- **Total Parameters**: ~234K (lightweight and efficient)

---

## ⚙️ Hyperparameters

Default hyperparameters in `src/config.py`:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `BATCH_SIZE` | 64 | Number of samples per batch |
| `LEARNING_RATE` | 0.001 | Adam optimizer learning rate |
| `N_EPOCHS` | 25 | Number of training epochs |
| `HIDDEN_SIZE` | 100 | LSTM hidden state size |
| `NUM_LAYERS` | 2 | Number of LSTM layers |
| `DROPOUT` | 0.5 | Dropout probability |
| `N_FEATURES` | 6 | Number of input channels |
| `SEQUENCE_LENGTH` | 128 | Length of time series window |
| `N_CLASSES` | 6 | Number of activity classes |

**To modify:** Edit values in `src/config.py` before training

---

## 📊 Expected Results

### Performance Metrics

With default hyperparameters, you should achieve:
- **Test Accuracy**: ~90-93%
- **Training Time**: 3-5 minutes (GPU) / 15-20 minutes (CPU)

### Common Confusions

The model typically confuses:
- SITTING ↔ STANDING (similar sensor readings when stationary)
- WALKING ↔ WALKING_UPSTAIRS (similar motion patterns)

These are visible in the confusion matrix visualization.

---

## 🐛 Troubleshooting

### Issue: "No module named 'src'"
**Solution:** Run commands with `-m` flag from project root:
```bash
python -m src.train  # NOT: python src/train.py
```

### Issue: "FileNotFoundError: UCI HAR Dataset"
**Solution:** Download the dataset first:
```bash
python data/download_dataset.py
```

### Issue: "CUDA out of memory"
**Solution:** Reduce batch size in `src/config.py`:
```python
BATCH_SIZE = 32  # or 16
```

### Issue: Training is very slow
**Solution:** 
- Check if CUDA is available: `python -c "import torch; print(torch.cuda.is_available())"`
- If False and you have GPU, reinstall PyTorch with CUDA support
- Otherwise, reduce epochs or batch size for faster testing

### Device/Environment controls
You can force CPU or disable cuDNN (workaround for certain GPU setups):
```bash
# Force CPU
HAR_FORCE_CPU=1 python -m src.train

# Use GPU but disable cuDNN for RNNs (avoids some version issues)
HAR_DISABLE_CUDNN=1 python -m src.train

# Explicitly set device
HAR_DEVICE=cuda python -m src.train
```

---

## 📝 For Your Report

Your 4-page report should include:

### Page 1: Introduction
- Problem statement (What is HAR? Why important?)
- Dataset description (UCI HAR, 30 subjects, 6 activities)
- Project objective

### Page 2: Methodology
- Data preprocessing
- Model architecture (CNN-LSTM) with diagram
- Justification for architecture choice
- Hyperparameters

### Page 3: Experiments and Results
- Training setup (optimizer, loss function, hyperparameters)
- Training curves (use `training_history.png`)
- Test results (classification report)
- Confusion matrix (use `confusion_matrix.png`)
- Analysis of results

### Page 4: Conclusion
- Summary of achievements
- Challenges faced
- Future improvements
- References

**Figures to include:**
1. Training/Validation curves (`saved_models/training_history.png`)
2. Confusion matrix (`saved_models/confusion_matrix.png`)
3. Model architecture diagram (draw manually or use visualization tools)
4. Sample signal plots (from data exploration notebook)

---

## 📚 Dataset Information

**Dataset:** Human Activity Recognition Using Smartphones  
**Source:** UCI Machine Learning Repository  
**Citation:**
```
Davide Anguita, Alessandro Ghio, Luca Oneto, Xavier Parra and Jorge L. Reyes-Ortiz. 
A Public Domain Dataset for Human Activity Recognition Using Smartphones. 
21th European Symposium on Artificial Neural Networks, Computational Intelligence 
and Machine Learning, ESANN 2013. Bruges, Belgium 24-26 April 2013.
```

**Details:**
- 30 volunteers (19-48 years)
- 6 activities performed
- Samsung Galaxy S II smartphone
- Accelerometer and gyroscope sensors
- 50 Hz sampling rate
- 2.56 second sliding windows (128 readings)
- 70/30 train/test split

---

## 🔄 Submission Checklist

Before submitting, ensure:

- [ ] All code runs without errors
- [ ] Dataset download instructions are clear
- [ ] Model can be trained successfully
- [ ] Evaluation produces results
- [ ] `requirements.txt` includes all dependencies
- [ ] README is complete with all instructions
- [ ] Saved model is included (or training instructions are clear)
- [ ] Project structure is organized
- [ ] Comments are added to code
- [ ] Report is prepared (max 4 pages)

---

## 🚀 Quick Start Guide

**Complete workflow in 5 steps:**

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download dataset
python data/download_dataset.py

# 3. Train model
python -m src.train

# 4. Evaluate model
python -m src.evaluate

# 5. Make predictions
python -m src.inference
```

**Total time:** ~20-30 minutes (including training)

---

## 📞 Support

If you encounter any issues:
1. Check the Troubleshooting section above
2. Verify all dependencies are installed correctly
3. Ensure dataset is downloaded properly
4. Check Python version (3.8+)

---

## 📄 License

This project is for educational purposes as part of the Deep Learning course at IIT Jodhpur.

---

## 🙏 Acknowledgments

- UCI Machine Learning Repository for providing the HAR dataset
- PyTorch team for the deep learning framework
- Course instructors and TAs for guidance

---

**Last Updated:** October 2025  
**Version:** 1.0.0
