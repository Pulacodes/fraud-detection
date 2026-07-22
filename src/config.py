"""Application configuration and constants for the fraud detection service."""

from __future__ import annotations

from pathlib import Path
from typing import Final

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
DATA_DIR: Final[Path] = PROJECT_ROOT / "data"
RAW_DATA_PATH: Final[Path] = DATA_DIR / "raw" / "creditcard.csv"
PROCESSED_DATA_DIR: Final[Path] = DATA_DIR / "processed"
TRAIN_FEATURES_PATH: Final[Path] = PROCESSED_DATA_DIR / "X_train.csv"
TRAIN_TARGET_PATH: Final[Path] = PROCESSED_DATA_DIR / "y_train.csv"
TEST_FEATURES_PATH: Final[Path] = PROCESSED_DATA_DIR / "X_test.csv"
TEST_TARGET_PATH: Final[Path] = PROCESSED_DATA_DIR / "y_test.csv"
MODEL_DIR: Final[Path] = PROJECT_ROOT / "models"
MODEL_PATH: Final[Path] = MODEL_DIR / "fraud_detection_pipeline.pkl"
RANDOM_STATE: Final[int] = 42
TARGET_COLUMN: Final[str] = "Class"
THRESHOLD: Final[float] = 0.5
FEATURE_ORDER: Final[list[str]] = [
    "Time",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
    "V7",
    "V8",
    "V9",
    "V10",
    "V11",
    "V12",
    "V13",
    "V14",
    "V15",
    "V16",
    "V17",
    "V18",
    "V19",
    "V20",
    "V21",
    "V22",
    "V23",
    "V24",
    "V25",
    "V26",
    "V27",
    "V28",
    "Amount",
]
SCALING_COLUMNS: Final[list[str]] = ["Time", "Amount"]
