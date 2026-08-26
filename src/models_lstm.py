"""Loading and running the pre-trained LSTM classification/regression models."""

import os

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

from src.config import (
    DISCHARGE_COL,
    LSTM_CLASSIFICATION_PATH,
    LSTM_REGRESSION_PATH,
    SEQUENCE_LENGTH,
)
from src.preprocessing import engineer_features, lstm_feature_columns


@st.cache_resource(show_spinner=False)
def load_lstm_models() -> dict | None:
    """Load both .h5 models once per session; None if the files aren't present."""
    if not (os.path.exists(LSTM_CLASSIFICATION_PATH) and os.path.exists(LSTM_REGRESSION_PATH)):
        return None
    return {
        "classification_model": tf.keras.models.load_model(LSTM_CLASSIFICATION_PATH),
        "regression_model": tf.keras.models.load_model(LSTM_REGRESSION_PATH),
    }


def _last_sequence(df: pd.DataFrame, sequence_length: int = SEQUENCE_LENGTH):
    data = engineer_features(df)
    features = lstm_feature_columns(data)
    if len(data) < sequence_length:
        return None, features
    sequence = data[features].values[-sequence_length:]
    return sequence.reshape(1, sequence_length, len(features)), features


def predict_next_step(df: pd.DataFrame, lstm_bundle: dict, sequence_length: int = SEQUENCE_LENGTH):
    """Single-step discharge and flood-probability prediction from the last known sequence."""
    X_pred, features = _last_sequence(df, sequence_length)
    if X_pred is None:
        return None, None, "Not enough data for a prediction (need at least 7 rows)."

    discharge_pred = float(lstm_bundle["regression_model"].predict(X_pred, verbose=0)[0][0])
    flood_prob = float(lstm_bundle["classification_model"].predict(X_pred, verbose=0)[0][0])
    return discharge_pred, flood_prob, None


def forecast_lstm(df: pd.DataFrame, lstm_bundle: dict, days: int, sequence_length: int = SEQUENCE_LENGTH) -> pd.DataFrame:
    """Roll the LSTM models forward day by day.

    Only the discharge slot in the feature sequence is updated between steps
    (the other engineered features — precip lags, seasonality — are not
    recomputed from a real future), so accuracy degrades the further out the
    forecast goes. Same limitation as the Random Forest path.
    """
    X_seq, features = _last_sequence(df, sequence_length)
    if X_seq is None:
        return pd.DataFrame(columns=["Date", "Discharge", "Flood_Probability"])

    discharge_idx = features.index(DISCHARGE_COL) if DISCHARGE_COL in features else 0
    last_date = df["Time"].max()
    current_sequence = X_seq
    results = []

    for step in range(days):
        discharge_pred = float(lstm_bundle["regression_model"].predict(current_sequence, verbose=0)[0][0])
        flood_prob = float(lstm_bundle["classification_model"].predict(current_sequence, verbose=0)[0][0])

        results.append({
            "Date": last_date + pd.Timedelta(days=step + 1),
            "Discharge": discharge_pred,
            "Flood_Probability": flood_prob,
        })

        next_sequence = np.roll(current_sequence[0], -1, axis=0)
        next_sequence[-1, discharge_idx] = discharge_pred
        current_sequence = next_sequence.reshape(1, sequence_length, len(features))

    return pd.DataFrame(results)
