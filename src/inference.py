# src/inference.py
"""
Inference script for making predictions on new data samples
"""
import torch
import numpy as np
from src.model import CnnLstmModel
import src.config as config

class HARPredictor:
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


def demo_inference():
    """
    Demo function showing how to use the predictor
    """
    print("="*60)
    print("HAR INFERENCE DEMO")
    print("="*60)
    
    # Load a test sample from the test dataset
    from src.data_loader import get_data_loaders
    
    _, test_loader = get_data_loaders(batch_size=1)
    
    # Get first sample
    signals, labels = next(iter(test_loader))
    sample_signal = signals[0].numpy()
    true_label = labels[0].item()
    
    activity_names = [
        "WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS", 
        "SITTING", "STANDING", "LAYING"
    ]
    
    print(f"\nTest Sample Shape: {sample_signal.shape}")
    print(f"True Activity: {activity_names[true_label]}")
    
    # Initialize predictor
    predictor = HARPredictor()
    
    # Make prediction
    print("\nMaking prediction...")
    result = predictor.predict(sample_signal)
    
    print("\n" + "-"*60)
    print("PREDICTION RESULTS")
    print("-"*60)
    print(f"Predicted Activity: {result['predicted_activity']}")
    print(f"Confidence: {result['confidence']:.4f}")
    print(f"\nAll Class Probabilities:")
    for activity, prob in result['all_probabilities'].items():
        print(f"  {activity:20s}: {prob:.4f}")
    
    print("\n" + "="*60)
    if result['predicted_activity'] == activity_names[true_label]:
        print("✓ CORRECT PREDICTION!")
    else:
        print("✗ INCORRECT PREDICTION")
    print("="*60)


if __name__ == "__main__":
    demo_inference()

