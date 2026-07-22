# Credit Card Fraud Detection

A production-ready machine learning project for detecting fraudulent credit card transactions using a scikit-learn pipeline and an XGBoost classifier.

## Overview

This project trains a binary classification model on transaction data and exposes it through a Streamlit UI for inference. The pipeline is designed to be reproducible, modular, and deployment-friendly.

## Features

- End-to-end training workflow from processed CSV files
- Reusable preprocessing and feature ordering logic
- Persisted model artifact using joblib
- Streamlit-based prediction interface
- Logging, validation, and error handling for production use
- Test suite for preprocessing, training, and prediction behavior

## Project Structure

- app/streamlit_app.py - Streamlit user interface
- data/raw/creditcard.csv - Raw dataset
- data/processed/ - Processed training and test CSV files
- src/config.py - Centralized configuration constants
- src/data.py - Data loading and validation utilities
- src/model.py - Training, persistence, and prediction logic
- src/train.py - Training entry point
- src/predict.py - Prediction entry point
- tests/ - Automated tests

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Training

```bash
python src/train.py
```

## Prediction

```bash
python src/predict.py
```

## Running the Streamlit App

```bash
streamlit run app/streamlit_app.py
```

## Testing

```bash
pytest
```

## Dataset

The project uses the Credit Card Fraud Detection dataset from kaggle with transaction features such as Time, Amount, and V1-V28, plus the target column Class.

## Future Improvements

- Add model evaluation metrics and confusion matrix reporting
- Introduce MLOps workflow with experiment tracking
- Add CI/CD automation and container deployment

## License

This project is provided for educational and demonstration purposes.
