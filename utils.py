import os
import logging
from pathlib import Path

# Paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
MODELS_DIR = PROJECT_DIR / "models"
APP_DIR = PROJECT_DIR / "app"
SRC_DIR = PROJECT_DIR / "src"

# Create directories if they don't exist
for directory in [DATA_DIR, MODELS_DIR, APP_DIR, SRC_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Configure logging
def setup_logging():
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.FileHandler(PROJECT_DIR / "project.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("JEE_NEET_Predictor")

logger = setup_logging()
