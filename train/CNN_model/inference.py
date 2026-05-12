import torch
from pathlib import Path
import numpy as np
from .model import ASLClassifier
from . import config

class SignLanguagePredictor:
    def __init__(self, model_path=None, device=None):
        self.device = device if device else config.DEVICE
        
        # Resolve paths using pathlib
        # config.py is in the same directory as this file (CNN_model pkg)
        package_dir = Path(__file__).parent.resolve()
        
        # classes.npy path
        # If config.LABEL_ENCODER_PATH is absolute, use it. 
        # If relative, assume it's relative to the package directory or project root.
        # Given the structure, it seems to be in CNN_model/classes.npy
        
        if Path(config.LABEL_ENCODER_PATH).is_absolute():
            label_path = Path(config.LABEL_ENCODER_PATH)
        else:
            # Try relative to package dir first (likely case for CNN_model/classes.npy)
            candidate = package_dir / config.LABEL_ENCODER_PATH
            if candidate.exists():
                label_path = candidate
            else:
                 # Fallback to current working directory (e.g. if running from root)
                 label_path = Path.cwd() / config.LABEL_ENCODER_PATH

        try:
            self.classes = np.load(label_path, allow_pickle=True)
            print(f"Loaded classes: {self.classes}")
        except FileNotFoundError:
            print(f"Warning: classes.npy not found at {label_path}")
            self.classes = []

        # Initialize Model
        self.model = ASLClassifier(config.INPUT_SIZE, config.NUM_CLASSES).to(self.device)
        
        # Load Weights
        if model_path:
             self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        else:
             # Fallback to default model path in config
             # Logic: Try relative to package, then CWD
             ckpt_name = config.MODEL_SAVE_PATH
             
             ckpt_candidates = [
                 package_dir / ckpt_name,
                 Path.cwd() / "CNN_model" / ckpt_name, # Common case if running from root
                 Path(ckpt_name)
             ]
             
             loaded = False
             for ckpt_path in ckpt_candidates:
                 if ckpt_path.exists():
                     print(f"Loading fallback model from: {ckpt_path}")
                     self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device))
                     loaded = True
                     break
            
             if not loaded:
                 print(f"Warning: No model weights found. Tried: {[str(p) for p in ckpt_candidates]}")
        
        self.model.eval()

    def predict(self, landmarks):
        """
        Predict from landmarks.
        Args:
            landmarks: np.array of shape (21, 3) or (63,)
        Returns:
            dict with 'predicted_class', 'confidence', 'all_probabilities'
        """
        # Flatten if needed
        if landmarks.ndim > 1:
            data = landmarks.flatten()
        else:
            data = landmarks.copy()
            
        # Add batch dim
        input_tensor = torch.tensor(data, dtype=torch.float32).to(self.device).unsqueeze(0)
        
        with torch.no_grad():
            output = self.model(input_tensor)
            probs = torch.softmax(output, dim=1)
            confidence, idx = probs.max(dim=1)
            
            idx = idx.item()
            confidence = confidence.item()
            
            predicted_class = self.classes[idx] if idx < len(self.classes) else "Unknown"
            
            all_probs = {self.classes[i]: probs[0][i].item() for i in range(len(self.classes))}
            
        return {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'all_probabilities': all_probs
        }