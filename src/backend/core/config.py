"""
Core application configuration and constants.
"""
from pathlib import Path

# Base paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
STATIC_MODEL_DIR = MODELS_DIR / "static"
DYNAMIC_MODEL_DIR = MODELS_DIR / "dynamic"
ASSETS_DIR = PROJECT_ROOT / "src" / "assets"
LETTER_GIFS_DIR = ASSETS_DIR  # GIFs are directly in assets folder

# Model paths
STATIC_MODEL_PATH = STATIC_MODEL_DIR / "best_model.pth"
DYNAMIC_MODEL_PATH = DYNAMIC_MODEL_DIR / "best_model.pth"
CLASSES_PATH = STATIC_MODEL_DIR / "classes.npy"

# Letter configuration
STATIC_LETTERS = list("ABCDEFGHIKLMNOPQRSTUVWXY")  # All except J, Z
DYNAMIC_LETTERS = ["J", "Z"]
ALL_LETTERS = sorted(STATIC_LETTERS + DYNAMIC_LETTERS)

# Detection settings - NEW WORKFLOW
RECORDING_DURATION = 3.0  # seconds to record for each attempt
STATIC_FPS = 10  # frames per second during static recording
DYNAMIC_FPS = 15  # frames per second during dynamic recording (needs 30 frames)
CONFIDENCE_THRESHOLD = 0.6  # Minimum confidence for valid prediction

# Session settings
MAX_ATTEMPT_TIME = 60  # seconds before timeout per letter
HINT_THRESHOLD_ATTEMPTS = 3  # Show hint after this many failed attempts
MAX_HINTS = 2  # Maximum number of hints per letter

# Dynamic letter buffer settings
DYNAMIC_BUFFER_SIZE = 30  # frames needed for LSTM

# WebSocket settings
WS_HEARTBEAT_INTERVAL = 30  # seconds
