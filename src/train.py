"""Training entry point for the fraud detection model."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import TRAIN_FEATURES_PATH, TRAIN_TARGET_PATH
from src.data import load_training_data
from src.model import train_model_from_dataframes

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    """Train the fraud detection pipeline and persist the model artifact."""
    logger.info("Starting training workflow")
    features, targets = load_training_data(
        features_path=TRAIN_FEATURES_PATH,
        target_path=TRAIN_TARGET_PATH,
    )
    train_model_from_dataframes(features, targets)
    logger.info("Training workflow completed successfully")


if __name__ == "__main__":
    main()
