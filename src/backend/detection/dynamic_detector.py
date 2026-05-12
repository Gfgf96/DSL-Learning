"""
Dynamic sign language detector for letters requiring movement (J, Z).
LSTM-based detection using trained model.
"""
import sys
from pathlib import Path
import torch
import numpy as np
from collections import deque

# Add parent to path if running standalone
if str(Path(__file__).parent.parent.parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.backend.models.lstm_model import DynamicSignLSTM, DynamicSignTransformer


class DynamicSignPredictor:
    """
    Predictor for dynamic ASL signs (J and Z) that require movement.
    Uses LSTM to analyze sequences of hand landmarks over time.
    """
    
    def __init__(self, model_path=None, device=None):
        """
        Initialize the dynamic sign predictor.
        
        Args:
            model_path: Path to trained LSTM model weights
            device: torch.device to run inference on
        """
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Set model path - default to models/dynamic/
        if model_path is None:
            project_root = Path(__file__).parent.parent.parent.parent
            model_path = project_root / "models" / "dynamic" / "best_model.pth"
        
        self.model_path = Path(model_path)
        self.model = None
        self.classes = ['J', 'Z']
        self.sequence_length = 30
        self.input_size = 63
        
        # Frame buffer for collecting sequences
        self.buffer = deque(maxlen=self.sequence_length)
        self.is_collecting = False
        
        # Load model
        self._load_model()
    
    def _load_model(self):
        """Load trained LSTM model."""
        if not self.model_path.exists():
            print(f"⚠ Dynamic model not found: {self.model_path}")
            print("   Dynamic letters (J, Z) will not work.")
            return
        
        try:
            # Load with weights_only=False for compatibility
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
            model_config = checkpoint.get('config', {})
            
            # Create model architecture
            model_type = model_config.get('model_type', 'lstm')
            if model_type == 'lstm':
                self.model = DynamicSignLSTM(
                    input_size=model_config.get('input_size', self.input_size),
                    hidden_size=model_config.get('hidden_size', 128),
                    num_layers=model_config.get('num_layers', 2),
                    num_classes=model_config.get('num_classes', 2)
                )
            else:
                self.model = DynamicSignTransformer(
                    input_size=model_config.get('input_size', self.input_size),
                    num_classes=model_config.get('num_classes', 2)
                )
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            val_acc = checkpoint.get('val_acc', checkpoint.get('accuracy', 'N/A'))
            print(f" Dynamic model loaded: {self.model_path.name}")
            if val_acc != 'N/A':
                print(f"   Validation accuracy: {val_acc:.3f}")
                
        except Exception as e:
            print(f" Error loading dynamic model: {e}")
            self.model = None
    
    def normalize_landmarks(self, landmarks):
        """
        Normalize landmarks (same as training).
        
        Args:
            landmarks: np.array of shape (21, 3) or (63,)
        Returns:
            Normalized flat array of shape (63,)
        """
        if landmarks.ndim == 1:
            points = landmarks.reshape(21, 3)
        else:
            points = landmarks.copy()
        
        # Center on wrist
        wrist = points[0].copy()
        points = points - wrist
        
        # Scale by hand size (distance to middle finger MCP)
        scale = np.linalg.norm(points[9])
        if scale > 0.001:
            points = points / scale
        
        return points.flatten()
    
    def start_collecting(self):
        """Start collecting frames for a dynamic gesture."""
        self.buffer.clear()
        self.is_collecting = True
    
    def stop_collecting(self):
        """Stop collecting frames."""
        self.is_collecting = False
    
    def add_frame(self, landmarks):
        """
        Add a frame to the buffer.
        
        Args:
            landmarks: np.array of shape (21, 3) or (63,)
        Returns:
            True if buffer is full and ready for prediction
        """
        if not self.is_collecting:
            return False
        
        # Normalize and add
        normalized = self.normalize_landmarks(landmarks)
        self.buffer.append(normalized)
        
        return len(self.buffer) == self.sequence_length
    
    def predict(self, landmark_sequence=None):
        """
        Predict dynamic sign from sequence of hand landmarks.
        
        Args:
            landmark_sequence: Optional np.array of shape (seq_len, 63)
                             If None, uses internal buffer
            
        Returns:
            dict with:
                - predicted_class: str, predicted letter
                - confidence: float, confidence score
                - all_probabilities: dict, probabilities for J and Z
        """
        if self.model is None:
            return {
                'predicted_class': 'Unknown',
                'confidence': 0.0,
                'all_probabilities': {'J': 0.0, 'Z': 0.0}
            }
        
        # Use provided sequence or buffer
        if landmark_sequence is not None:
            seq = landmark_sequence
        else:
            if len(self.buffer) < 10:  # Minimum frames
                return None
            seq = np.array(list(self.buffer), dtype=np.float32)
        
        # Interpolate to sequence_length if needed
        if len(seq) != self.sequence_length:
            seq = self._interpolate_sequence(seq)
        
        # Convert to tensor
        X = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(self.device)  # (1, 30, 63)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(X)
            probs = torch.softmax(outputs, dim=1)[0]
            pred_idx = probs.argmax().item()
            confidence = probs[pred_idx].item()
        
        predicted_class = self.classes[pred_idx]
        all_probs = {self.classes[i]: probs[i].item() for i in range(len(self.classes))}
        
        return {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'all_probabilities': all_probs
        }
    
    def _interpolate_sequence(self, seq):
        """Interpolate sequence to target length."""
        try:
            from scipy.interpolate import interp1d
            x_old = np.linspace(0, 1, len(seq))
            x_new = np.linspace(0, 1, self.sequence_length)
            seq_interp = np.zeros((self.sequence_length, 63))
            for i in range(63):
                f = interp1d(x_old, seq[:, i], kind='linear')
                seq_interp[:, i] = f(x_new)
            return seq_interp.astype(np.float32)
        except ImportError:
            # Fallback: simple repetition or truncation
            if len(seq) > self.sequence_length:
                return seq[:self.sequence_length]
            else:
                # Repeat last frame
                padding = np.tile(seq[-1], (self.sequence_length - len(seq), 1))
                return np.vstack([seq, padding]).astype(np.float32)
    
    def get_buffer_progress(self):
        """Get current buffer fill percentage (0-1)."""
        return len(self.buffer) / self.sequence_length
    
    def clear_buffer(self):
        """Clear the frame buffer."""
        self.buffer.clear()
