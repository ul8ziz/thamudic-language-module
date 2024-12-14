from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
LETTERS_DIR = DATA_DIR / "letters"
PROCESSED_DIR = DATA_DIR / "processed"

# Model configuration
MODEL_CONFIG = {
    "image_size": 128,
    "batch_size": 32,
    "learning_rate": 0.001,
    "num_epochs": 100,
    "early_stopping_patience": 10,
    "validation_split": 0.2,
}

# Data augmentation configuration
AUGMENTATION_CONFIG = {
    "rotation_range": 15,
    "width_shift_range": 0.1,
    "height_shift_range": 0.1,
    "zoom_range": 0.1,
    "shear_range": 0.1,
}

# Preprocessing configuration
PREPROCESSING_CONFIG = {
    "target_size": (128, 128),
    "grayscale": True,
    "normalize": True,
    "denoise_strength": 10,
    "contrast_limit": 3.0,
    "brightness_limit": 0.3,
}

# Training configuration
TRAINING_CONFIG = {
    "optimizer": "adam",
    "loss": "categorical_crossentropy",
    "metrics": ["accuracy"],
    "checkpoint_dir": MODELS_DIR / "checkpoints",
    "tensorboard_dir": MODELS_DIR / "tensorboard",
}

# Character segmentation configuration
SEGMENTATION_CONFIG = {
    "min_contour_area": 100,
    "min_aspect_ratio": 0.2,
    "max_aspect_ratio": 5.0,
    "padding": 5,
    "direction": "rtl",  # right-to-left for Arabic/Thamudic
}

# GUI configuration
GUI_CONFIG = {
    "window_title": "Thamudic Script Recognition",
    "window_size": "800x600",
    "max_display_size": 400,
    "theme": "default",
}
