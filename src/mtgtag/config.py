"""Configuration management for MTGTag pipeline."""

import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

# Data file paths
TAG_DEFINITIONS_PATH = DATA_DIR / "tag_definitions.json"
FULL_CARD_DATABASE_PATH = DATA_DIR / "full_card_database.csv"
FUNCTIONAL_TAGS_PATH = DATA_DIR / "functional_tags.json"

# Model paths
DOMAIN_ADAPTED_MODEL_PATH = MODELS_DIR / "domain_adapted"
CLASSIFIER_MODEL_PATH = MODELS_DIR / "classifier"

# Training configuration
DEFAULT_CONFIG = {
    "domain_adaptation": {
        "base_model": "distilbert-base-uncased",
        "epochs": 3,
        "batch_size": 16,
        "learning_rate": 5e-5,
        "max_length": 512
    },
    "classifier_training": {
        "epochs": 3,
        "batch_size": 16,
        "learning_rate": 2e-5,
        "validation_split": 0.2
    },
    "threshold_optimization": {
        "metric": "f1",
        "search_range": (0.1, 0.9),
        "search_steps": 50
    }
}