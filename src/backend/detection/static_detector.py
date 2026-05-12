"""
Static sign language detector using CNN for landmark classification.
Upgraded to use ResidualMLP architecture with enhanced checkpoint loading.
"""
import torch
import numpy as np
import pickle
from pathlib import Path
from src.backend.models.cnn_model import ResidualMLP
from src.backend.models import config


class StaticSignPredictor:
    """
    Predictor for static ASL signs (all letters except J and Z).
    Uses a trained ResidualMLP to classify hand landmarks.
    """
    
    def __init__(self, model_path=None, device=None):
        """
        Initialize the static sign predictor.
        
        Args:
            model_path: Path to trained model weights (.pth file)
            device: torch.device to run inference on
        """
        self.device = device if device else config.DEVICE
        self.model_path = Path(model_path) if model_path else config.MODEL_SAVE_PATH
        
        # Load class labels (try both .npy and .pkl formats)
        self.classes = self._load_classes()
        
        # Load model
        self.model = self._load_model()
        self.model.eval()
    
    def _load_classes(self):
        """Load class labels from .npy or .pkl file."""
        model_dir = self.model_path.parent
        
        # Try classes.npy first
        npy_path = model_dir / "classes.npy"
        if npy_path.exists():
            try:
                classes = np.load(npy_path, allow_pickle=True)
                print(f"✓ Loaded {len(classes)} classes from classes.npy")
                return classes
            except Exception as e:
                print(f"Warning: Failed to load classes.npy: {e}")
        
        # Try label_encoder.pkl
        pkl_path = model_dir / "label_encoder.pkl"
        if pkl_path.exists():
            try:
                with open(pkl_path, 'rb') as f:
                    le = pickle.load(f)
                classes = le.classes_
                print(f"✓ Loaded {len(classes)} classes from label_encoder.pkl")
                return classes
            except Exception as e:
                print(f"Warning: Failed to load label_encoder.pkl: {e}")
        
        print(f"Warning: No class labels found in {model_dir}")
        return []
    
    def _load_model(self):
        """Load model with support for both old and new checkpoint formats."""
        if not self.model_path.exists():
            print(f"Warning: Model not found at {self.model_path}")
            return ResidualMLP(config.INPUT_SIZE, config.NUM_CLASSES).to(self.device)
        
        try:
            # Load with weights_only=False for compatibility with older checkpoints
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
            
            # Check if it's a new-style checkpoint with metadata
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                # New format with metadata
                num_classes = checkpoint.get('num_classes', config.NUM_CLASSES)
                input_dim = checkpoint.get('input_dim', config.INPUT_SIZE)
                model_name = checkpoint.get('model_name', 'ResidualMLP')
                test_acc = checkpoint.get('test_acc', None)
                
                model = ResidualMLP(
                    input_dim=input_dim,
                    num_classes=num_classes,
                    hidden_dim=256,
                    num_blocks=4,
                    dropout=0.3
                )
                model.load_state_dict(checkpoint['model_state_dict'])
                model.to(self.device)
                
                print(f"✓ Loaded {model_name} from {self.model_path.name}")
                if test_acc is not None:
                    print(f"  Test accuracy: {test_acc:.2f}%")
            else:
                # Old format - just state_dict
                num_classes = len(self.classes) if self.classes else config.NUM_CLASSES
                model = ResidualMLP(
                    input_dim=config.INPUT_SIZE,
                    num_classes=num_classes
                )
                model.load_state_dict(checkpoint)
                model.to(self.device)
                print(f"✓ Loaded model (legacy format) from {self.model_path.name}")
            
            return model
            
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Creating new model with default parameters...")
            return ResidualMLP(config.INPUT_SIZE, config.NUM_CLASSES).to(self.device)

    def predict(self, landmarks):
        """
        Predict sign from hand landmarks.
        
        Args:
            landmarks: np.array of shape (21, 3) or (63,) - hand landmarks
            
        Returns:
            dict with:
                - predicted_class: str, predicted letter
                - confidence: float, confidence score for prediction
                - all_probabilities: dict, all class probabilities
        """
        # Flatten if needed
        if landmarks.ndim > 1:
            data = landmarks.flatten()
        else:
            data = landmarks.copy()
            
        # Convert to tensor and add batch dimension
        input_tensor = torch.tensor(data, dtype=torch.float32).to(self.device).unsqueeze(0)
        
        with torch.no_grad():
            output = self.model(input_tensor)
            probs = torch.softmax(output, dim=1)
            confidence, idx = probs.max(dim=1)
            
            idx = idx.item()
            confidence = confidence.item()
            
            predicted_class = self.classes[idx] if idx < len(self.classes) else "Unknown"
            
            # Get probabilities for all classes
            all_probs = {self.classes[i]: probs[0][i].item() for i in range(len(self.classes))}
            
        return {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'all_probabilities': all_probs
        }
