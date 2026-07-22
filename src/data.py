"""Data loading and feature preparation utilities for the fraud detection pipeline."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import (
    FEATURE_ORDER,
    TARGET_COLUMN,
    TEST_FEATURES_PATH,
    TEST_TARGET_PATH,
    TRAIN_FEATURES_PATH,
    TRAIN_TARGET_PATH,
)

logger = logging.getLogger(__name__)


def _require_columns(dataframe: pd.DataFrame, expected_columns: list[str], *, context: str) -> None:
    missing = [column for column in expected_columns if column not in dataframe.columns]
    if missing:
        raise ValueError(f"Missing required columns for {context}: {', '.join(missing)}")


def prepare_features(
    dataframe: pd.DataFrame,
    *,
    feature_order: list[str] | None = None,
) -> pd.DataFrame:
    """Prepare a feature DataFrame with a consistent order and required columns.

    Args:
        dataframe: Raw feature data.
        feature_order: Optional ordered list of feature names to enforce.

    Returns:
        A DataFrame with consistent feature ordering and a copy of the input data.
    """
    if dataframe.empty:
        raise ValueError("Input dataframe is empty.")

    requested_features = list(feature_order or FEATURE_ORDER)
    ordered_features = list(dict.fromkeys([*requested_features, *FEATURE_ORDER]))

    prepared = dataframe.copy()
    for column in ordered_features:
        if column not in prepared.columns:
            prepared[column] = 0.0

    prepared = prepared.loc[:, ordered_features].copy()
    prepared = prepared.astype(float)
    return prepared


def load_training_data(
    *,
    features_path: Path | None = None,
    target_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Load and validate the training feature and target data."""
    features_path = features_path or TRAIN_FEATURES_PATH
    target_path = target_path or TRAIN_TARGET_PATH

    features = pd.read_csv(features_path)
    targets = pd.read_csv(target_path)

    if targets.columns.size != 1:
        raise ValueError("Training target file must contain exactly one column.")

    target_column = targets.columns[0]
    if target_column != TARGET_COLUMN:
        targets = targets.rename(columns={target_column: TARGET_COLUMN})

    features = prepare_features(features)
    targets = targets[TARGET_COLUMN].astype(int)

    return features, targets


def load_test_data(
    *,
    features_path: Path | None = None,
    target_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Load and validate the test feature and target data."""
    features_path = features_path or TEST_FEATURES_PATH
    target_path = target_path or TEST_TARGET_PATH

    features = pd.read_csv(features_path)
    targets = pd.read_csv(target_path)

    if targets.columns.size != 1:
        raise ValueError("Test target file must contain exactly one column.")

    target_column = targets.columns[0]
    if target_column != TARGET_COLUMN:
        targets = targets.rename(columns={target_column: TARGET_COLUMN})

    features = prepare_features(features)
    targets = targets[TARGET_COLUMN].astype(int)

    return features, targets


def validate_uploaded_csv(uploaded_file: Any) -> pd.DataFrame:
    """Validate an uploaded CSV file and return it as a DataFrame."""
    if uploaded_file is None:
        raise ValueError("No file was provided.")

    if not hasattr(uploaded_file, "name") or not uploaded_file.name.endswith(".csv"):
        raise ValueError("Please upload a CSV file.")

    try:
        dataframe = pd.read_csv(uploaded_file)
    except Exception as exc:  # pragma: no cover - exercised through runtime validation
        raise ValueError(f"Unable to read the uploaded CSV: {exc}") from exc

    if dataframe.empty:
        raise ValueError("Uploaded CSV is empty.")

    feature_frame = prepare_features(dataframe)
    return feature_frame
