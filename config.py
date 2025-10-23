# config.py

# Data
DATA_PATH = "nslkdd_data"
K_NEIGHBORS = 5
NUM_CLIENTS = 3

# Model
HIDDEN_CHANNELS = 32
DROPOUT = 0.3
OUT_CHANNELS = 2
HEADS = 4

# Training
ROUNDS = 6
LOCAL_EPOCHS = 3
LR = 0.005

# Differential Privacy
USE_DP = True
CLIP_NORM = 1.0
NOISE_MULTIPLIER = 0.5
DELTA = 1e-5

# Device
DEVICE = "cuda"  # set to "cpu" if no GPU
