"""
Configuration for CNN-based static sign language classifier.
Migrated from CNN_model/config.py
"""
import torch
from pathlib import Path

# Base directory for model files
# Points to models/static/ where best_model.pth and classes.npy are stored
BASE_DIR = Path(__file__).parent.parent.parent.parent / "models" / "static"

# Data Settings
DATA_PATH = 'ngt.npz'
MODEL_SAVE_PATH = BASE_DIR / 'best_model.pth'
LABEL_ENCODER_PATH = BASE_DIR / 'classes.npy'

# Model Hyperparameters
INPUT_SIZE = 63  # 21 landmarks * 3 coordinates (x, y, z)
NUM_CLASSES = 25  # A-Z excluding J and Z (24 letters) + Nonsense class = 25
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 0.001

# Training Settings
PATIENCE = 7
MIN_DELTA = 0.001
LR_PATIENCE = 3
LR_FACTOR = 0.5
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
RANDOM_SEED = 42

# Inference Settings
CONFIDENCE_THRESHOLD = 0.8

# Hardware
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
