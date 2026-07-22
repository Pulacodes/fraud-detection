"""Prediction entry point for the fraud detection model."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model import load_model_artifact, predict_from_dataframe

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    """Run a sample prediction using the persisted model artifact."""
    sample = pd.DataFrame(
        {
            "Time": [1000.0],
            "V1": [0.0],
            "V2": [0.0],
            "V3": [0.0],
            "V4": [0.0],
            "V5": [0.0],
            "V6": [0.0],
            "V7": [0.0],
            "V8": [0.0],
            "V9": [0.0],
            "V10": [0.0],
            "V11": [0.0],
            "V12": [0.0],
            "V13": [0.0],
            "V14": [0.0],
            "V15": [0.0],
            "V16": [0.0],
            "V17": [0.0],
            "V18": [0.0],
            "V19": [0.0],
            "V20": [0.0],
            "V21": [0.0],
            "V22": [0.0],
            "V23": [0.0],
            "V24": [0.0],
            "V25": [0.0],
            "V26": [0.0],
            "V27": [0.0],
            "V28": [0.0],
            "Amount": [250.0],
        }
    )

    artifact = load_model_artifact()
    result = predict_from_dataframe(sample, artifact)
    logger.info("Sample prediction: %s", result)


if __name__ == "__main__":
    main()
