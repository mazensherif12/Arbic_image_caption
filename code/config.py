"""
Configuration settings
"""

import os

# ========== PATHS ==========
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data paths
IMAGE_DIR = os.path.join(ROOT, "data/Flickr8k_Dataset")
FEATURES_FILE = os.path.join(ROOT, 'data/vgg16_features.pickle')
DATA_FILE = os.path.join(ROOT, "data/Flickr8k_text", "Flickr8k.arabic.full.txt")
TEST_IMGS_FILE = os.path.join(ROOT, "data/Flickr8k_text", "Flickr_8k.testImages.txt")

# Model paths
MODEL_PATH = os.path.join(ROOT, "trained_model.keras")
CHECKPOINT_DIR = os.path.join(ROOT, "checkpoints")
RESULTS_DIR = os.path.join(ROOT, "test_results")

# Create directories
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ========== CONSTANTS ==========
START_TOKEN = "<START>"
END_TOKEN = "<END>"
PAD_TOKEN = "<PAD>"
MAXLEN = 20

# ========== TRAINING SETTINGS ==========
NUM_TRAIN_IMAGES = 3000
NUM_TEST_IMAGES = 500
NUM_EPOCHS = 25
BATCH_SIZE = 64
LEARNING_RATE = 0.0003

if __name__ == "__main__":
    print("Configuration loaded:")
    print(f"ROOT: {ROOT}")
    print(f"IMAGE_DIR: {IMAGE_DIR}")
    print(f"MODEL_PATH: {MODEL_PATH}")