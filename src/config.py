"""
تكوين المشروع - تحديد المسارات والإعدادات
"""

from pathlib import Path

# المسارات الرئيسية
PROJECT_ROOT = Path(__file__).parent.parent
SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RUNS_DIR = PROJECT_ROOT / "runs"

# مسارات البيانات
RAW_DATA_DIR = DATA_DIR / "raw"
LETTERS_DIR = DATA_DIR / "letters"
MAPPING_FILE = DATA_DIR / "mapping.json"

# مسارات النماذج
MODEL_CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"
MODEL_CONFIGS_DIR = MODELS_DIR / "configs"
BEST_MODEL_PATH = MODEL_CHECKPOINTS_DIR / "best_model.pt"

# إعدادات النموذج
MODEL_CONFIG = {
    "image_size": 224,
    "batch_size": 32,
    "learning_rate": 0.001,
    "num_epochs": 100,
    "early_stopping_patience": 10,
    "validation_split": 0.2,
}

# إعدادات معالجة الصور
IMAGE_PROCESSING_CONFIG = {
    "contrast_limit": 2.0,
    "brightness_limit": 1.2,
    "sharpness_limit": 1.5,
    "rotation_range": 15,
    "width_shift_range": 0.1,
    "height_shift_range": 0.1,
    "zoom_range": 0.1,
    "shear_range": 0.1,
}

# إعدادات التدريب
TRAINING_CONFIG = {
    "optimizer": "adamw",
    "loss": "cross_entropy",
    "metrics": ["accuracy"],
    "scheduler": "cosine_annealing",
    "min_lr": 1e-6,
    "weight_decay": 1e-4,
}
