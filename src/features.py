"""Feature engineering and scaling helpers for the fraud detection project."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.config import MODEL_DIR

logger = logging.getLogger(__name__)


def fit_scalers(features: pd.DataFrame) -> tuple[StandardScaler, StandardScaler, pd.DataFrame]:
    """Fit scalers on the provided feature data and return transformed features."""
    amount_scaler = StandardScaler()
    time_scaler = StandardScaler()

    transformed_features = features.copy()
    transformed_features["Amount"] = amount_scaler.fit_transform(transformed_features[["Amount"]])
    transformed_features["Time"] = time_scaler.fit_transform(transformed_features[["Time"]])

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(amount_scaler, MODEL_DIR / "amount_scaler.pkl")
    joblib.dump(time_scaler, MODEL_DIR / "time_scaler.pkl")
    logger.info("Saved feature scalers to %s", MODEL_DIR)
    return amount_scaler, time_scaler, transformed_features
