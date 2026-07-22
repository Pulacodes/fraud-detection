"""Streamlit frontend for interacting with the fraud detection model."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import MODEL_PATH, THRESHOLD
from src.data import validate_uploaded_csv
from src.model import load_model_artifact, predict_from_dataframe, save_prediction_report

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Fraud Detection", page_icon="💳", layout="wide")


@st.cache_resource(show_spinner=False)
def load_artifact() -> dict[str, Any]:
    """Load the serialized model artifact with Streamlit caching."""
    return load_model_artifact(MODEL_PATH)


st.title("Credit Card Fraud Detection")
st.markdown("Upload transaction features or enter a single transaction to estimate fraud risk.")

threshold = st.sidebar.slider("Fraud Threshold", 0.0, 1.0, THRESHOLD, 0.01)
uploaded_file = st.sidebar.file_uploader("Upload transactions CSV", type=["csv"])

if uploaded_file is not None:
    try:
        dataframe = validate_uploaded_csv(uploaded_file)
        artifact = load_artifact()
        result = predict_from_dataframe(dataframe, artifact)

        st.success("Prediction completed successfully.")
        st.dataframe(dataframe.head(), use_container_width=True)
        st.metric("Prediction", "Fraud" if result["prediction"] == 1 else "Legitimate")
        st.metric("Fraud Probability", f"{result['probability']:.2%}")

        if st.button("Save prediction report"):
            report_path = MODEL_PATH.parent / "prediction_report.json"
            save_prediction_report(result, report_path)
            st.success(f"Report saved to {report_path}")
    except Exception as exc:  # pragma: no cover - runtime feedback path
        logger.exception("Prediction failed")
        st.error(f"Prediction failed: {exc}")
else:
    st.info("Please upload a CSV file to begin.")
