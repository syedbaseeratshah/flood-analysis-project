"""Random Forest training and multi-day discharge forecasting."""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

from src.config import DISCHARGE_COL, PRECIP_COL
from src.preprocessing import engineer_features, rf_feature_columns


def train_random_forest(df: pd.DataFrame, test_fraction: float = 0.2) -> dict:
    """Fit a Random Forest on engineered features, held out on a trailing slice."""
    data = engineer_features(df).dropna()
    features = rf_feature_columns(data)

    X, y = data[features], data[DISCHARGE_COL]
    split = int(len(X) * (1 - test_fraction))
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    metrics = {
        "mae": mean_absolute_error(y_test, y_pred),
        "r2": r2_score(y_test, y_pred),
    }

    return {"model": model, "scaler": scaler, "features": features, "metrics": metrics}


def forecast_random_forest(df: pd.DataFrame, rf_bundle: dict, days: int) -> pd.DataFrame:
    """Roll the trained RF model forward day by day.

    Future precipitation is unknown, so it's held at zero for forecast steps
    (a standard no-forecast-input simplification) — the further out the
    forecast, the more this understates any precipitation-driven flood risk.
    """
    model, scaler, features = rf_bundle["model"], rf_bundle["scaler"], rf_bundle["features"]

    working = df.copy()
    last_date = working["Time"].max()
    predictions = []

    for step in range(days):
        next_date = last_date + pd.Timedelta(days=step + 1)
        engineered = engineer_features(working)
        latest = engineered.iloc[-1:].copy()
        latest["Time"] = next_date
        latest["Month"] = next_date.month
        latest["Month_sin"] = np.sin(2 * np.pi * next_date.month / 12)
        latest["Month_cos"] = np.cos(2 * np.pi * next_date.month / 12)

        X_pred = scaler.transform(latest[features])
        discharge_pred = float(model.predict(X_pred)[0])

        latest[DISCHARGE_COL] = discharge_pred
        latest[PRECIP_COL] = 0.0
        working = pd.concat([working, latest[df.columns]], ignore_index=True)

        predictions.append({"Date": next_date, "Discharge": discharge_pred})

    return pd.DataFrame(predictions)
