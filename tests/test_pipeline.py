from pathlib import Path

import pandas as pd

from src.data import prepare_features
from src.model import build_pipeline, predict_from_dataframe, train_model_from_dataframes


def test_prepare_features_reorders_and_validates_columns() -> None:
    df = pd.DataFrame(
        {
            "Amount": [10.0, 20.0],
            "Time": [100.0, 200.0],
            "V1": [0.1, -0.2],
            "V2": [0.2, -0.1],
            "V3": [0.3, -0.3],
            "V4": [0.4, -0.4],
            "V5": [0.5, -0.5],
            "V6": [0.6, -0.6],
            "V7": [0.7, -0.7],
            "V8": [0.8, -0.8],
            "V9": [0.9, -0.9],
            "V10": [1.0, -1.0],
            "V11": [1.1, -1.1],
            "V12": [1.2, -1.2],
            "V13": [1.3, -1.3],
            "V14": [1.4, -1.4],
            "V15": [1.5, -1.5],
            "V16": [1.6, -1.6],
            "V17": [1.7, -1.7],
            "V18": [1.8, -1.8],
            "V19": [1.9, -1.9],
            "V20": [2.0, -2.0],
            "V21": [2.1, -2.1],
            "V22": [2.2, -2.2],
            "V23": [2.3, -2.3],
            "V24": [2.4, -2.4],
            "V25": [2.5, -2.5],
            "V26": [2.6, -2.6],
            "V27": [2.7, -2.7],
            "V28": [2.8, -2.8],
        }
    )

    prepared = prepare_features(df, feature_order=["Time", "Amount", "V1", "V2"])

    assert list(prepared.columns[:4]) == ["Time", "Amount", "V1", "V2"]
    assert prepared.shape[0] == 2


def test_train_model_from_dataframes_creates_artifact(tmp_path: Path) -> None:
    X = pd.DataFrame(
        {
            "Time": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "Amount": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            "V1": [0.1, -0.2, 0.3, -0.4, 0.5, -0.6],
            "V2": [0.2, -0.1, 0.2, -0.1, 0.2, -0.1],
            "V3": [0.3, -0.3, 0.4, -0.4, 0.5, -0.5],
            "V4": [0.4, -0.4, 0.5, -0.5, 0.6, -0.6],
        }
    )
    y = pd.Series([0, 0, 1, 0, 1, 0], name="Class")

    artifact_path = tmp_path / "model.joblib"
    artifact = train_model_from_dataframes(X, y, artifact_path)

    assert artifact_path.exists()
    assert artifact["pipeline"] is not None
    assert artifact["feature_names"][:4] == ["Time", "V1", "V2", "V3"]


def test_predict_from_dataframe_returns_probability_and_label() -> None:
    X = pd.DataFrame(
        {
            "Time": [0.0],
            "Amount": [10.0],
            "V1": [0.1],
            "V2": [0.2],
            "V3": [0.3],
            "V4": [0.4],
        }
    )
    training_frame = pd.DataFrame(
        {
            "Time": [0.0, 1.0, 2.0, 3.0],
            "Amount": [10.0, 20.0, 30.0, 40.0],
            "V1": [0.1, -0.2, 0.3, -0.4],
            "V2": [0.2, -0.1, 0.2, -0.1],
            "V3": [0.3, -0.3, 0.4, -0.4],
            "V4": [0.4, -0.4, 0.5, -0.5],
        }
    )
    pipeline = build_pipeline()
    pipeline.fit(training_frame, [0, 0, 1, 1])

    artifact = {
        "pipeline": pipeline,
        "feature_names": list(X.columns),
        "threshold": 0.5,
    }

    result = predict_from_dataframe(X, artifact)

    assert set(result.keys()) == {"prediction", "probability"}
    assert result["prediction"] in {0, 1}
    assert 0.0 <= result["probability"] <= 1.0
