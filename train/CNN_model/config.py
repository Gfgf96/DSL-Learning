import torch

# Data Settings
DATA_PATH = 'ngt.npz'
MODEL_SAVE_PATH = 'best_model.pth'
LABEL_ENCODER_PATH = 'classes.npy'

# Model Hyperparameters
INPUT_SIZE = 63
NUM_CLASSES = 24
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