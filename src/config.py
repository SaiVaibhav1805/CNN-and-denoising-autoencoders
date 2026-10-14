"""Configuration and Hyperparameters for Smart Waste Classification.

PRD Specifications:
- 6-class waste categorization: cardboard, glass, metal, paper, plastic, trash
- Input resolution: 128 x 128 x 3 (normalized to [0, 1])
- Denoising Autoencoder & CNN Classifier
"""

import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
SPLITS_DIR = DATA_DIR / "splits"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

# Ensure directories exist
for directory in [DATA_DIR, RAW_DATA_DIR, SPLITS_DIR, MODELS_DIR, REPORTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Dataset Classes (TrashNet standard 6 classes)
CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
NUM_CLASSES = len(CLASSES)
CLASS_TO_IDX = {cls_name: i for i, cls_name in enumerate(CLASSES)}
IDX_TO_CLASS = {i: cls_name for i, cls_name in enumerate(CLASSES)}

# Image & Model Specifications
IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128
CHANNELS = 3
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS)

# Reproducibility
RANDOM_SEED = 42

# Data Splitting
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Noise Parameters for Denoising Autoencoder
NOISE_FACTOR = 0.25  # Gaussian noise standard deviation factor
NOISE_TYPE = "gaussian"  # 'gaussian' or 'salt_and_pepper'

# Training Hyperparameters
BATCH_SIZE = 32
CNN_EPOCHS = 25
CNN_LEARNING_RATE = 1e-3
DAE_EPOCHS = 30
DAE_LEARNING_RATE = 1e-3

# Model Save Paths
CNN_MODEL_PATH = MODELS_DIR / "cnn_classifier.keras"
DAE_MODEL_PATH = MODELS_DIR / "denoising_autoencoder.keras"
ENCODER_MODEL_PATH = MODELS_DIR / "encoder.keras"
SPLIT_TRAIN_FILE = SPLITS_DIR / "train.json"
SPLIT_VAL_FILE = SPLITS_DIR / "val.json"
SPLIT_TEST_FILE = SPLITS_DIR / "test.json"
EVALUATION_REPORT_FILE = REPORTS_DIR / "evaluation_metrics.json"
