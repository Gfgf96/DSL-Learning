"""Detection module for sign language recognition.

Handles hand capture, preprocessing, and sign classification.
"""
from src.backend.detection.hand_capture import HandCapture, normalize
from src.backend.detection.static_detector import StaticSignPredictor
from src.backend.detection.dynamic_detector import DynamicSignPredictor
from src.backend.detection.sign_detector import SignDetector, DYNAMIC_LETTERS

__all__ = [
    'HandCapture',
    'normalize',
    'StaticSignPredictor',
    'DynamicSignPredictor',
    'SignDetector',
    'DYNAMIC_LETTERS'
]
