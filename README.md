# Deep Learning Project: Human Activity Recognition

This project implements a Deep Learning model (CNN-LSTM) in PyTorch to classify human activities from smartphone sensor data.

**Course:** Deep Learning (CS-XXX)
**Degree:** M.Tech Data Science and Data Engineering, IIT Jodhpur
**Group Members:** [Your Name(s)]

---

## 📋 Project Overview

This project implements a **hybrid CNN-LSTM deep learning model** in PyTorch to classify human activities from smartphone sensor data. The model achieves high accuracy in distinguishing between six different activities using raw accelerometer and gyroscope readings.

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
│   ├── download_dataset.py          # Script to download UCI HAR Dataset
│   └── UCI HAR Dataset/             # Dataset folder (created after download)
│
├── notebooks/
│   └── 01_data_exploration.ipynb    # Data exploration and visualization
│
├── saved_models/
│   ├── best_model.pth               # Trained model weights (created after training)
│   ├── training_history.png         # Training/validation curves (created after training)
│   └── confusion_matrix.png         # Confusion matrix visualization (created after evaluation)
│
├── src/
│   ├── __init__.py                  # Package initialization
│   ├── config.py                    # Configuration and hyperparameters
│   ├── data_loader.py               # PyTorch Dataset and DataLoader
│   ├── model.py                     # CNN-LSTM model architecture
│   ├── train.py                     # Training script
│   ├── evaluate.py                  # Evaluation script
│   └── inference.py                 # Inference script for predictions
│
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
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

Run inference demo on a sample:

```bash
python -m src.inference
```

**What happens:**
- Loads a test sample
- Makes prediction using trained model
- Shows prediction with confidence scores for all classes

**Use in Your Code:**
```python
from src.inference import HARPredictor

# Initialize predictor
predictor = HARPredictor("./saved_models/best_model.pth")

# Make prediction on single sample (shape: 6, 128)
result = predictor.predict(signal_data)

print(f"Predicted: {result['predicted_activity']}")
print(f"Confidence: {result['confidence']:.2f}")
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
- `HARPredictor` class for making predictions
- Supports single sample and batch prediction
- Returns predictions with confidence scores
- Demo function included

**`data/download_dataset.py`**
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
