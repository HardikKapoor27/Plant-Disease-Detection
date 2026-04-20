"""
config.py - Central configuration for Plant Disease Detection Project
"""

import os

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DATA_DIR        = os.path.join(BASE_DIR, "data")
TRAIN_DIR       = os.path.join(DATA_DIR, "train")
VALID_DIR       = os.path.join(DATA_DIR, "valid")
MODEL_DIR       = os.path.join(BASE_DIR, "saved_models")
PLOT_DIR        = os.path.join(BASE_DIR, "plots")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR,  exist_ok=True)

# ─── Dataset ──────────────────────────────────────────────────────────────────
# 14-class subset of PlantVillage (much smaller than full 38-class dataset)
SELECTED_CLASSES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___healthy",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___healthy",
]

NUM_CLASSES = len(SELECTED_CLASSES)

# Friendly display names for the classes
CLASS_DISPLAY_NAMES = {
    "Apple___Apple_scab":               "Apple - Apple Scab",
    "Apple___Black_rot":                "Apple - Black Rot",
    "Apple___healthy":                  "Apple - Healthy",
    "Corn_(maize)___Common_rust_":      "Corn - Common Rust",
    "Corn_(maize)___healthy":           "Corn - Healthy",
    "Grape___Black_rot":                "Grape - Black Rot",
    "Grape___healthy":                  "Grape - Healthy",
    "Potato___Early_blight":            "Potato - Early Blight",
    "Potato___Late_blight":             "Potato - Late Blight",
    "Potato___healthy":                 "Potato - Healthy",
    "Tomato___Bacterial_spot":          "Tomato - Bacterial Spot",
    "Tomato___Early_blight":            "Tomato - Early Blight",
    "Tomato___Late_blight":             "Tomato - Late Blight",
    "Tomato___healthy":                 "Tomato - Healthy",
}

# ─── Training Hyperparameters ─────────────────────────────────────────────────
IMAGE_SIZE   = 224          # Input image dimensions (224×224)
BATCH_SIZE   = 32
NUM_EPOCHS   = 20
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
MAX_LR       = 0.01         # For OneCycleLR scheduler

# ─── Images per class (subset size) ──────────────────────────────────────────
# Original PlantVillage has 1000-2000 images/class; we cap at 300 for speed
MAX_IMAGES_PER_CLASS = 300  # Change to None to use all available images

# ─── Saved model path ─────────────────────────────────────────────────────────
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, "plant_disease_model.pth")
HISTORY_SAVE_PATH = os.path.join(MODEL_DIR, "training_history.pkl")

# ─── Random seed ──────────────────────────────────────────────────────────────
SEED = 42
