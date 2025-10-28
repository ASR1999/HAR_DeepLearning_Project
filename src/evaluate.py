# src/evaluate.py
import torch
from src.data_loader import get_data_loaders
from src.model import CnnLstmModel
import src.config as config
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_confusion_matrix(cm, class_names, save_path="./saved_models/confusion_matrix.png"):
    """Plot and save confusion matrix as a heatmap"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, 
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix - HAR Model', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix plot saved to {save_path}")
    plt.close()

def evaluate_model():
    print("="*60)
    print("HAR MODEL EVALUATION")
    print("="*60)
    
    # Create saved_models directory if it doesn't exist
    os.makedirs("./saved_models", exist_ok=True)
    
    print("\nLoading test data...")
    _, test_loader = get_data_loaders(config.BATCH_SIZE)
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Load the saved model
    print("\nLoading saved model from './saved_models/best_model.pth'...")
    model = CnnLstmModel(
        n_features=config.N_FEATURES,
        n_classes=config.N_CLASSES,
        hidden_size=config.HIDDEN_SIZE,
        num_layers=config.NUM_LAYERS,
        dropout=0.0  # Set dropout to 0 for evaluation
    ).to(config.DEVICE)
    
    model.load_state_dict(torch.load("./saved_models/best_model.pth", map_location=config.DEVICE))
    model.eval()
    print("✓ Model loaded successfully")
    
    all_labels = []
    all_preds = []
    
    print("\nRunning inference on test set...")
    with torch.no_grad():
        for signals, labels in test_loader:
            signals, labels = signals.to(config.DEVICE), labels.to(config.DEVICE)
            
            outputs = model(signals)
            _, predicted = torch.max(outputs.data, 1)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            
    # Calculate metrics
    print("\n" + "="*60)
    print("TEST RESULTS")
    print("="*60)
    
    activity_names = [
        "WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS", 
        "SITTING", "STANDING", "LAYING"
    ]
    
    # Calculate overall accuracy
    accuracy = 100 * np.sum(np.array(all_labels) == np.array(all_preds)) / len(all_labels)
    print(f"\nOverall Test Accuracy: {accuracy:.2f}%\n")
    
    print("\nClassification Report:")
    print("-"*60)
    print(classification_report(all_labels, all_preds, target_names=activity_names, digits=4))
    
    print("\nConfusion Matrix (Raw Counts):")
    print("-"*60)
    cm = confusion_matrix(all_labels, all_preds)
    print(cm)
    
    # Plot confusion matrix
    plot_confusion_matrix(cm, activity_names)
    
    print("\n" + "="*60)
    print("Evaluation completed successfully!")
    print("="*60)

if __name__ == "__main__":
    evaluate_model()