"""Feature engineering shared by the Random Forest and LSTM prediction paths.

Both models were previously fed by near-identical, hand-duplicated blocks in
app.py and flood_prediction_app.py. Keeping one version here means a fix or
tweak only has to happen once, and the two models are guaranteed to see the
same feature definitions.
"""

import numpy as np
import pandas as pd
import streamlit as st

from src.config import DISCHARGE_COL, LAG_STEPS, PRECIP_COL, SOIL_MOISTURE_COL


@st.cache_data(show_spinner=False)
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add lag, rolling, seasonal, and interaction features used by both models."""
    data = df.set_index("Time").copy()

    for col in data.columns:
        data[col] = data[col].interpolate(method="time", limit=3)
    data = data.ffill()

    data["Month"] = data.index.month
    data["Month_sin"] = np.sin(2 * np.pi * data["Month"] / 12)
    data["Month_cos"] = np.cos(2 * np.pi * data["Month"] / 12)

    for lag in LAG_STEPS:
        data[f"Precip_lag_{lag}"] = data[PRECIP_COL].shift(lag)
        data[f"Discharge_lag_{lag}"] = data[DISCHARGE_COL].shift(lag)

    data["Precip_sum_7d"] = data[PRECIP_COL].rolling(window=7).sum()
    data["Discharge_mean_7d"] = data[DISCHARGE_COL].rolling(window=7).mean()

    if SOIL_MOISTURE_COL in data.columns:
        data["Precip_x_SM"] = data[PRECIP_COL] * data[SOIL_MOISTURE_COL] / 100

    data = data.ffill().bfill()
    return data.reset_index()


def lstm_feature_columns(data: pd.DataFrame) -> list[str]:
    """Feature subset the pre-trained LSTM models were trained on."""
    features = [
        DISCHARGE_COL, PRECIP_COL, "PET(mm h^-1)",
        "Discharge_lag_1", "Discharge_lag_3", "Precip_lag_1", "Precip_lag_7",
        "Precip_sum_7d", "Discharge_mean_7d", "Month_sin", "Month_cos",
    ]
    if SOIL_MOISTURE_COL in data.columns:
        features += [SOIL_MOISTURE_COL, "Precip_x_SM"]
    return [col for col in features if col in data.columns]


def rf_feature_columns(data: pd.DataFrame) -> list[str]:
    """Feature subset used to train the Random Forest model."""
    features = [
        PRECIP_COL, "PET(mm h^-1)",
        "Precip_lag_1", "Precip_lag_3", "Precip_lag_7",
        "Discharge_lag_1", "Discharge_lag_2", "Discharge_lag_3",
        "Precip_sum_7d", "Discharge_mean_7d",
        "Month_sin", "Month_cos",
    ]
    if SOIL_MOISTURE_COL in data.columns:
        features.append(SOIL_MOISTURE_COL)
    if "Groundwater (mm)" in data.columns:
        features.append("Groundwater (mm)")
    return [col for col in features if col in data.columns]


def flood_threshold(df: pd.DataFrame, percentile: float) -> float:
    return df[DISCHARGE_COL].quantile(percentile)
