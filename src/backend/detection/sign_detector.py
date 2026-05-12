"""
Unified sign detector that handles both static and dynamic signs.
Based on src/backend/detector.py with improved architecture.
"""
import numpy as np
from src.backend.detection.hand_capture import HandCapture, normalize
from src.backend.detection.static_detector import StaticSignPredictor
from src.backend.detection.dynamic_detector import DynamicSignPredictor

# Letters that require movement detection
DYNAMIC_LETTERS = ['J', 'Z']


class SignDetector:
    """
    Main detector class that routes to appropriate predictor based on letter type.
    Handles both static (CNN) and dynamic (LSTM) sign detection.
    """
    
    def __init__(self, static_model_path=None, dynamic_model_path=None, device=None):
        """
        Initialize the sign detector.
        
        Args:
            static_model_path: Path to CNN model for static signs
            dynamic_model_path: Path to LSTM model for dynamic signs
            device: torch.device for inference
        """
        self.hand_capture = HandCapture()
        self.buffer = []  # For accumulating frames for dynamic detection
        
        # Initialize static predictor (CNN)
        self.static_predictor = None
        if static_model_path:
            try:
                self.static_predictor = StaticSignPredictor(static_model_path, device=device)
                print(f"Static detector initialized with model: {static_model_path}")
            except Exception as e:
                print(f"Failed to load static model: {e}")
        
        # Initialize dynamic predictor (LSTM)
        self.dynamic_predictor = None
        if dynamic_model_path:
            try:
                self.dynamic_predictor = DynamicSignPredictor(dynamic_model_path, device=device)
                print(f"Dynamic detector initialized with model: {dynamic_model_path}")
            except Exception as e:
                print(f"Failed to load dynamic model: {e}")
        
    def is_dynamic(self, letter):
        """Check if letter requires movement (dynamic detection)."""
        return letter.upper() in DYNAMIC_LETTERS

    def process_frame(self, frame, target_letter):
        """
        Process a single frame and detect sign language gesture.
        
        Args:
            frame: BGR image from camera
            target_letter: str, the letter user is trying to sign
            
        Returns:
            tuple: (landmarks, status, data, prediction_info)
                - landmarks: raw landmarks array or None
                - status: str, human-readable status message
                - data: prepared data for model (if ready) or None
                - prediction_info: dict with prediction results or None
        """
        landmarks = self.hand_capture.extract_landmarks(frame)
        
        status = f"Letter: {target_letter} | Hand: NO"
        data = None
        prediction_info = None
        
        if landmarks is not None:
            norm = normalize(landmarks)
            status = f"Letter: {target_letter} | Hand: YES"
            
            if self.is_dynamic(target_letter):
                # === Dynamic Letter Logic (J, Z) ===
                self.buffer.append(norm)
                
                # Keep buffer at max 30 frames
                if len(self.buffer) > 30:
                    self.buffer.pop(0)
                
                status += f" | Buffer: {len(self.buffer)}/30"
                
                if len(self.buffer) == 30:
                    data_array = np.array(self.buffer)
                    data = data_array 
                    status += f" | Ready: {data.shape}"
                    
                    # TODO: Use dynamic predictor when LSTM is ready
                    if self.dynamic_predictor:
                        prediction_info = self.dynamic_predictor.predict(data, target_letter)
                        status += f" | Pred: {prediction_info['predicted_class']}"
                    
            else:
                # === Static Letter Logic (A, B, C, etc.) ===
                self.buffer = []  # Clear buffer for static detection
                
                if self.static_predictor:
                    try:
                        # Predict using CNN
                        result = self.static_predictor.predict(landmarks)
                        predicted_class = result['predicted_class']
                        confidence = result['confidence']
                        
                        status += f" | Pred: {predicted_class} ({confidence:.2f})"
                        prediction_info = result
                        
                        # Check if prediction matches target
                        if predicted_class == target_letter:
                            status += " [MATCH]"
                    except Exception as e:
                        print(f"Prediction Error: {e}")
                        status += " | Err"
                else:
                    # No model loaded, just return normalized data
                    data = norm.flatten()

        else:
            # No hand detected - keep buffer for dynamic if needed
            pass
            
        return landmarks, status, data, prediction_info
    
    def clear_buffer(self):
        """Clear the frame buffer (useful when changing target letter)."""
        self.buffer = []
