"""
LSTM Model Configuration for J/Z Dynamic Letters
"""

import torch
from pathlib import Path
import os

# Paths
# Default relative path inside the repository (near the train package). Allow
# override via `DSL_DYNAMIC_DATA` environment variable for CI / developer machines.
DEFAULT_DATA = (
    Path(__file__).parent.parent
    / "data_collect"
    / "ngt_dynamic"
    / "jz_dynamic_normalized.npz"
)
DATA_PATH = Path(os.environ.get("DSL_DYNAMIC_DATA", DEFAULT_DATA))
MODEL_DIR = Path(__file__).parent
CHECKPOINT_PATH = MODEL_DIR / "best_model.pth"

# Model Architecture
INPUT_SIZE = 63  # landmarks per frame (21 points × 3 coords)
SEQUENCE_LENGTH = 30  # frames per sequence
HIDDEN_SIZE = 128  # LSTM hidden units
NUM_LAYERS = 2  # LSTM layers
NUM_CLASSES = 2  # J and Z
DROPOUT = 0.3

# Training
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 100
PATIENCE = 15  # early stopping patience
TEST_SPLIT = 0.2

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Class mapping
CLASSES = ["J", "Z"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CLASS = {i: c for i, c in enumerate(CLASSES)}
