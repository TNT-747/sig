"""
Configuration file for Hand Gesture Recognition System
Contains all hyperparameters, paths, and constants
"""

import os

# ============================================================================
# PATHS
# ============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = BASE_DIR
TRAIN_DATA_PATH = os.path.join(DATA_DIR, 'sign_mnist_train.csv')
TEST_DATA_PATH = os.path.join(DATA_DIR, 'sign_mnist_test.csv')

# Model save directory
MODEL_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, 'hand_gesture_model.h5')
BEST_MODEL_PATH = os.path.join(MODEL_DIR, 'best_model.h5')

# Results directory
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================================================================
# DATA PARAMETERS
# ============================================================================
IMG_HEIGHT = 28
IMG_WIDTH = 28
IMG_CHANNELS = 1
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

# Number of classes (A-Y, excluding J and Z which require motion)
# Dataset has 24 unique labels spanning indices 0-24 (index 9 is not used)
NUM_CLASSES = 25

# Gesture labels mapping (0-24 -> A-Y, index 9 unused)
GESTURE_LABELS = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I',
    # 9 is not used (J requires motion and is excluded)
    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R',
    18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y'
}

# ============================================================================
# TRAINING HYPERPARAMETERS
# ============================================================================
BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.15

# Early stopping patience
EARLY_STOPPING_PATIENCE = 7

# Model checkpoint
SAVE_BEST_ONLY = True
MONITOR_METRIC = 'val_accuracy'

# ============================================================================
# DATA AUGMENTATION PARAMETERS
# ============================================================================
AUGMENTATION_PARAMS = {
    'rotation_range': 10,
    'zoom_range': 0.1,
    'width_shift_range': 0.1,
    'height_shift_range': 0.1,
    'horizontal_flip': False,  # Don't flip for sign language
    'fill_mode': 'nearest'
}

# ============================================================================
# CAMERA/PREDICTION PARAMETERS
# ============================================================================
CAMERA_ID = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# ROI (Region of Interest) for hand detection
ROI_TOP = 100
ROI_BOTTOM = 400
ROI_LEFT = 300
ROI_RIGHT = 600

# Prediction confidence threshold
CONFIDENCE_THRESHOLD = 0.6

# Display parameters
FONT = 1  # cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 1.0
FONT_THICKNESS = 2
TEXT_COLOR = (0, 255, 0)  # Green
BOX_COLOR = (0, 255, 0)   # Green

# ============================================================================
# MODEL ARCHITECTURE PARAMETERS
# ============================================================================
CNN_FILTERS = [32, 64, 128]
KERNEL_SIZE = (3, 3)
POOL_SIZE = (2, 2)
DROPOUT_RATE = 0.5
DENSE_UNITS = 128
