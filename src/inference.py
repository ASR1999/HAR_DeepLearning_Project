# src/inference.py
"""
Inference script for making predictions on new data samples.

Supports two modes:
- signals: CNN-LSTM on raw inertial windows (6×128) using best_model.pth
- features: MLP on 561-dim features (UCI/HAPT/combined) using best_mlp_*.pth
"""
import argparse
import torch
import numpy as np
from src.model import CnnLstmModel
from src.model_mlp import MLPClassifier
from src.data_loader_features import get_feature_loaders
import src.config as config

class SignalPredictor:
    """
    A predictor class for Human Activity Recognition
    """
    def __init__(self, model_path="./saved_models/best_model.pth"):
        """
        Initialize the predictor with a trained model
        
        Args:
            model_path: Path to the saved model weights
        """
        self.device = config.DEVICE
        self.model = CnnLstmModel(
            n_features=config.N_FEATURES,
            n_classes=config.N_CLASSES,
            hidden_size=config.HIDDEN_SIZE,
            num_layers=config.NUM_LAYERS,
            dropout=0.0  # No dropout for inference
        ).to(self.device)
        
        # Load the trained weights
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        self.activity_names = [
            "WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS", 
            "SITTING", "STANDING", "LAYING"
        ]
        
        print(f"Model loaded successfully from {model_path}")
        print(f"Using device: {self.device}")
    
    def predict(self, signal_data):
        """
        Make prediction on a single sample or batch of samples
        
        Args:
            signal_data: numpy array of shape (n_features, sequence_length) for single sample
                        or (batch_size, n_features, sequence_length) for batch
                        
        Returns:
            Dictionary containing:
                - predicted_class: class index (0-5)
                - predicted_activity: activity name
                - confidence: prediction confidence (0-1)
                - all_probabilities: probabilities for all classes
        """
        # Convert to tensor
        if len(signal_data.shape) == 2:
            # Single sample - add batch dimension
            signal_data = np.expand_dims(signal_data, axis=0)
        
        signal_tensor = torch.tensor(signal_data, dtype=torch.float32).to(self.device)
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(signal_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        # Convert to numpy
        predicted_class = predicted.cpu().numpy()[0]
        confidence_score = confidence.cpu().numpy()[0]
        all_probs = probabilities.cpu().numpy()[0]
        
        result = {
            'predicted_class': int(predicted_class),
            'predicted_activity': self.activity_names[predicted_class],
            'confidence': float(confidence_score),
            'all_probabilities': {
                name: float(prob) 
                for name, prob in zip(self.activity_names, all_probs)
            }
        }
        
        return result
    
    def predict_batch(self, signal_batch):
        """
        Make predictions on a batch of samples
        
        Args:
            signal_batch: numpy array of shape (batch_size, n_features, sequence_length)
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        signal_tensor = torch.tensor(signal_batch, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(signal_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidences, predicted = torch.max(probabilities, 1)
        
        # Convert to numpy
        predicted_classes = predicted.cpu().numpy()
        confidence_scores = confidences.cpu().numpy()
        all_probs = probabilities.cpu().numpy()
        
        for i in range(len(predicted_classes)):
            result = {
                'predicted_class': int(predicted_classes[i]),
                'predicted_activity': self.activity_names[predicted_classes[i]],
                'confidence': float(confidence_scores[i]),
                'all_probabilities': {
                    name: float(prob) 
                    for name, prob in zip(self.activity_names, all_probs[i])
                }
            }
            results.append(result)
        
        return results


class FeaturePredictor:
    """Predictor for 561-dim feature models (MLP)."""
    def __init__(self, input_dim: int, n_classes: int, model_path: str = "./saved_models/best_mlp_combined.pth"):
        self.device = config.DEVICE
        self.activity_names = [
            "WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS",
            "SITTING", "STANDING", "LAYING"
        ][:n_classes]
        self.model = MLPClassifier(input_dim=input_dim, n_classes=n_classes, dropout=0.0).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        print(f"MLP model loaded from {model_path} (input_dim={input_dim}, n_classes={n_classes})")

    def predict(self, feature_vector: np.ndarray):
        # feature_vector: (n_features,) or (batch, n_features)
        if feature_vector.ndim == 1:
            feature_vector = np.expand_dims(feature_vector, axis=0)
        x = torch.tensor(feature_vector, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)
            conf, pred = torch.max(probs, 1)
        pred_idx = int(pred.cpu().numpy()[0])
        conf_val = float(conf.cpu().numpy()[0])
        all_probs = probs.cpu().numpy()[0]
        return {
            'predicted_class': pred_idx,
            'predicted_activity': self.activity_names[pred_idx],
            'confidence': conf_val,
            'all_probabilities': {name: float(p) for name, p in zip(self.activity_names, all_probs)}
        }


def demo_inference():
    """
    Demo function showing how to use the predictor
    """
    print("="*60)
    print("HAR INFERENCE DEMO")
    print("="*60)
    
    parser = argparse.ArgumentParser(description="HAR inference (signals or features)")
    parser.add_argument("--mode", choices=["signals", "features"], default="signals")
    parser.add_argument("--model_path", type=str, default=None, help="Path to weights. Auto default per mode if not set")
    parser.add_argument("--source", choices=["uci", "hapt", "combined"], default="combined", help="Feature source for features mode")
    args = parser.parse_args()

    activity_names = [
        "WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS",
        "SITTING", "STANDING", "LAYING"
    ]

    if args.mode == "signals":
        # Load a test sample from raw inertial signals
        from src.data_loader import get_data_loaders
        _, test_loader = get_data_loaders(batch_size=1)
        signals, labels = next(iter(test_loader))
        sample_signal = signals[0].numpy()
        true_label = labels[0].item()
        print(f"\nTest Sample Shape: {sample_signal.shape}")
        print(f"True Activity: {activity_names[true_label]}")
        model_path = args.model_path or "./saved_models/best_model.pth"
        predictor = SignalPredictor(model_path=model_path)
        print("\nMaking prediction...")
        result = predictor.predict(sample_signal)
    else:
        # Load a test sample from features loader
        train_loader, test_loader, n_classes = get_feature_loaders(args.source, batch_size=1)
        X, y = next(iter(test_loader))
        x_feat = X[0].numpy()
        true_label = int(y[0].item())
        print(f"\nFeature vector shape: {x_feat.shape}")
        print(f"True Activity: {activity_names[true_label] if true_label < len(activity_names) else true_label}")
        model_path = args.model_path or f"./saved_models/best_mlp_{args.source}.pth"
        predictor = FeaturePredictor(input_dim=x_feat.shape[0], n_classes=n_classes, model_path=model_path)
        print("\nMaking prediction...")
        result = predictor.predict(x_feat)
    
    print("\n" + "-"*60)
    print("PREDICTION RESULTS")
    print("-"*60)
    print(f"Predicted Activity: {result['predicted_activity']}")
    print(f"Confidence: {result['confidence']:.4f}")
    print(f"\nAll Class Probabilities:")
    for activity, prob in result['all_probabilities'].items():
        print(f"  {activity:20s}: {prob:.4f}")
    
    print("\n" + "="*60)
    if result['predicted_activity'] == (activity_names[true_label] if true_label < len(activity_names) else str(true_label)):
        print("✓ CORRECT PREDICTION!")
    else:
        print("✗ INCORRECT PREDICTION")
    print("="*60)


if __name__ == "__main__":
    demo_inference()

