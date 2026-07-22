"""Model training, persistence, and inference for the fraud detection service."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.config import MODEL_PATH, PROJECT_ROOT, RANDOM_STATE, THRESHOLD, TARGET_COLUMN
from src.data import prepare_features

logger = logging.getLogger(__name__)


def build_pipeline() -> Pipeline:
    """Create the production model pipeline."""
    preprocessor = ColumnTransformer(
        transformers=[("scale", StandardScaler(), ["Time", "Amount"])],
        remainder="passthrough",
    )
    classifier = XGBClassifier(
        random_state=RANDOM_STATE,
        eval_metric="logloss",
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
    )
    return Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])


def train_model_from_dataframes(
    features: pd.DataFrame,
    targets: pd.Series,
    artifact_path: Path | str | None = None,
) -> dict[str, Any]:
    """Train a fraud detection pipeline and persist it to disk."""
    prepared_features = prepare_features(features)
    prepared_targets = targets.astype(int)

    if prepared_targets.name != TARGET_COLUMN:
        prepared_targets = prepared_targets.rename(TARGET_COLUMN)

    pipeline = build_pipeline()
    pipeline.fit(prepared_features, prepared_targets)

    artifact_path = Path(artifact_path or MODEL_PATH)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)

    artifact = {
        "pipeline": pipeline,
        "feature_names": list(prepared_features.columns),
        "threshold": THRESHOLD,
    }
    joblib.dump(artifact, artifact_path)
    logger.info("Training completed and model saved to %s", artifact_path)
    return artifact


def _resolve_artifact_path(artifact_path: Path | str | None = None) -> Path:
    """Resolve and validate a model artifact path."""
    resolved_path = Path(artifact_path or MODEL_PATH)
    if not resolved_path.is_absolute():
        resolved_path = (Path.cwd() / resolved_path).resolve()
    else:
        resolved_path = resolved_path.resolve()

    if not resolved_path.exists():
        raise FileNotFoundError(f"Model artifact not found at {resolved_path}")
    if not resolved_path.is_file():
        raise ValueError(f"Model artifact path is not a file: {resolved_path}")
    try:
        resolved_path.relative_to(PROJECT_ROOT)
    except ValueError as exc:
        raise ValueError("Model artifact path must remain within the project directory.") from exc
    return resolved_path


def load_model_artifact(artifact_path: Path | str | None = None) -> dict[str, Any]:
    """Load a persisted model artifact from disk."""
    artifact_path = _resolve_artifact_path(artifact_path)

    artifact = joblib.load(artifact_path)
    if not isinstance(artifact, dict):
        raise TypeError("Model artifact must be a dictionary.")

    if "pipeline" not in artifact or "feature_names" not in artifact:
        raise ValueError("Model artifact is missing required metadata.")

    logger.info("Loaded model artifact from %s", artifact_path)
    return artifact


def predict_from_dataframe(
    features: pd.DataFrame,
    artifact: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate a prediction and probability for a feature DataFrame."""
    artifact = artifact or load_model_artifact()
    prepared_features = prepare_features(features, feature_order=list(artifact["feature_names"]))

    pipeline = artifact["pipeline"]
    if pipeline is None:
        raise ValueError("Model artifact does not contain a trained pipeline.")

    probability = float(pipeline.predict_proba(prepared_features)[0][1])
    prediction = int(probability >= float(artifact.get("threshold", THRESHOLD)))
    return {"prediction": prediction, "probability": probability}


def save_prediction_report(payload: dict[str, Any], file_path: Path | str) -> None:
    """Persist a prediction payload as JSON."""
    output_path = Path(file_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
