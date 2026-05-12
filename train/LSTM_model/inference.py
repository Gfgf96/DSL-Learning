"""
Inference for LSTM Dynamic Sign Model (J/Z)
"""

import sys
from pathlib import Path

# Add parent directory to path for direct execution
SCRIPT_DIR = Path(__file__).parent
if str(SCRIPT_DIR.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR.parent))

import torch
import numpy as np
from collections import deque

from LSTM_model.model import DynamicSignLSTM, DynamicSignTransformer
from LSTM_model import config


class DynamicSignPredictor:
    """
    Predictor for dynamic sign language letters (J/Z).
    Collects frames and predicts when sequence is complete.
    """
    
    def __init__(self, model_path=None, device=None):
        self.device = device or config.DEVICE
        self.model_path = Path(model_path) if model_path else config.CHECKPOINT_PATH
        
        # Frame buffer
        self.buffer = deque(maxlen=config.SEQUENCE_LENGTH)
        self.is_collecting = False
        
        # Load model
        self.model = None
        self.classes = config.CLASSES
        self._load_model()
    
    def _load_model(self):
        """Load trained model."""
        if not self.model_path.exists():
            print(f"⚠️ Model not found: {self.model_path}")
            return
        
        checkpoint = torch.load(self.model_path, map_location=self.device)
        model_config = checkpoint.get('config', {})
        
        # Create model
        model_type = model_config.get('model_type', 'lstm')
        if model_type == 'lstm':
            self.model = DynamicSignLSTM(
                input_size=model_config.get('input_size', config.INPUT_SIZE),
                hidden_size=model_config.get('hidden_size', config.HIDDEN_SIZE),
                num_layers=model_config.get('num_layers', config.NUM_LAYERS),
                num_classes=model_config.get('num_classes', config.NUM_CLASSES)
            )
        else:
            self.model = DynamicSignTransformer(
                input_size=model_config.get('input_size', config.INPUT_SIZE),
                num_classes=model_config.get('num_classes', config.NUM_CLASSES)
            )
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"✅ Loaded model: {self.model_path}")
        print(f"   Accuracy: {checkpoint.get('val_acc', 'N/A'):.3f}")
    
    def normalize_landmarks(self, coords):
        """Normalize landmarks (same as training)."""
        points = coords.reshape(21, 3)
        wrist = points[0].copy()
        points = points - wrist
        scale = np.linalg.norm(points[9])
        if scale > 0.001:
            points = points / scale
        return points.flatten()
    
    def start_collecting(self):
        """Start collecting frames for a dynamic gesture."""
        self.buffer.clear()
        self.is_collecting = True
    
    def stop_collecting(self):
        """Stop collecting and return prediction."""
        self.is_collecting = False
        if len(self.buffer) >= 10:  # minimum frames
            return self.predict()
        return None
    
    def add_frame(self, landmarks):
        """
        Add a frame to the buffer.
        landmarks: numpy array of shape (63,)
        Returns prediction if buffer is full, else None.
        """
        if not self.is_collecting:
            return None
        
        # Normalize
        normalized = self.normalize_landmarks(landmarks)
        self.buffer.append(normalized)
        
        # Predict when buffer is full
        if len(self.buffer) == config.SEQUENCE_LENGTH:
            return self.predict()
        
        return None
    
    def predict(self):
        """
        Predict from current buffer.
        Returns: dict with 'letter', 'confidence', 'probabilities'
        """
        if self.model is None:
            return None
        
        if len(self.buffer) < 10:
            return None
        
        # Prepare sequence
        seq = np.array(list(self.buffer), dtype=np.float32)
        
        # Interpolate to SEQUENCE_LENGTH if needed
        if len(seq) != config.SEQUENCE_LENGTH:
            from scipy.interpolate import interp1d
            x_old = np.linspace(0, 1, len(seq))
            x_new = np.linspace(0, 1, config.SEQUENCE_LENGTH)
            seq_interp = np.zeros((config.SEQUENCE_LENGTH, 63))
            for i in range(63):
                f = interp1d(x_old, seq[:, i], kind='linear')
                seq_interp[:, i] = f(x_new)
            seq = seq_interp.astype(np.float32)
        
        # To tensor
        X = torch.tensor(seq).unsqueeze(0).to(self.device)  # (1, 30, 63)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(X)
            probs = torch.softmax(outputs, dim=1)[0]
            pred_idx = probs.argmax().item()
            confidence = probs[pred_idx].item()
        
        return {
            'letter': self.classes[pred_idx],
            'confidence': confidence,
            'probabilities': {
                self.classes[i]: probs[i].item() 
                for i in range(len(self.classes))
            }
        }
    
    def get_buffer_progress(self):
        """Get current buffer fill percentage."""
        return len(self.buffer) / config.SEQUENCE_LENGTH
    
    def clear_buffer(self):
        """Clear the frame buffer."""
        self.buffer.clear()


def predict_from_file(npz_path, model_path=None):
    """
    Predict all samples from a file.
    For testing purposes.
    """
    predictor = DynamicSignPredictor(model_path)
    
    data = np.load(npz_path, allow_pickle=True)
    X = data['X']
    y = data['y']
    
    correct = 0
    for i, (seq, label) in enumerate(zip(X, y)):
        # Simulate frame-by-frame input
        predictor.start_collecting()
        for frame in seq:
            result = predictor.add_frame(frame)
        
        if result is None:
            result = predictor.predict()
        
        if result and result['letter'] == label:
            correct += 1
        
        predictor.clear_buffer()
    
    accuracy = correct / len(X)
    print(f"Accuracy: {accuracy:.4f} ({correct}/{len(X)})")
    return accuracy


if __name__ == "__main__":
    # Test prediction
    predict_from_file(config.DATA_PATH)