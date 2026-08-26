"""Pure data computations behind each analysis tab (no Streamlit/plotting here)."""

import pandas as pd
import streamlit as st

from src.config import DISCHARGE_COL, MAX_MEMORY_LAG_DAYS, PET_COL, PRECIP_COL, SOIL_MOISTURE_COL


def flag_flood_events(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Add is_flood / flood_group columns identifying contiguous flood spells."""
    out = df.copy()
    out["is_flood"] = out[DISCHARGE_COL] > threshold
    out["flood_group"] = (out["is_flood"] != out["is_flood"].shift()).cumsum()
    return out


def summarize_flood_events(df: pd.DataFrame, threshold: float, min_duration: int = 2) -> list[dict]:
    flagged = flag_flood_events(df, threshold)
    events = []
    for group, spell in flagged[flagged["is_flood"]].groupby("flood_group"):
        if len(spell) < min_duration:
            continue
        events.append({
            "start": spell["Time"].min(),
            "end": spell["Time"].max(),
            "duration": len(spell),
            "peak": spell[DISCHARGE_COL].max(),
            "peak_date": spell.loc[spell[DISCHARGE_COL].idxmax(), "Time"],
            "total_precip": spell[PRECIP_COL].sum(),
            "group": group,
        })
    return sorted(events, key=lambda e: e["peak"], reverse=True)


@st.cache_data(show_spinner=False)
def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    columns = [DISCHARGE_COL, PRECIP_COL, PET_COL]
    for optional in [SOIL_MOISTURE_COL, "Groundwater (mm)", "Fast Flow(mm*1000)", "Slow Flow(mm*1000)", "Base Flow(mm*1000)"]:
        if optional in df.columns:
            columns.append(optional)

    working = df.copy()
    for lag in (1, 3, 7):
        col = f"Precip_lag_{lag}"
        working[col] = working[PRECIP_COL].shift(lag)
        columns.append(col)

    return working[columns].dropna().corr()


@st.cache_data(show_spinner=False)
def basin_memory_curve(df: pd.DataFrame, max_lag: int = MAX_MEMORY_LAG_DAYS) -> pd.DataFrame:
    """Correlation between discharge and lagged precipitation, for lags 1..max_lag.

    Vectorized with a single concat of shifted columns rather than mutating
    the dataframe in a Python loop — noticeably faster once max_lag climbs
    past a handful of days on a multi-year record.
    """
    lagged = pd.concat(
        {lag: df[PRECIP_COL].shift(lag) for lag in range(1, max_lag + 1)}, axis=1
    )
    correlations = lagged.corrwith(df[DISCHARGE_COL])
    return pd.DataFrame({"Lag (days)": correlations.index, "Correlation": correlations.values})


def memory_half_life(memory_curve: pd.DataFrame) -> tuple[int, float, float | None]:
    """Return (lag at peak correlation, peak correlation, lag where it first halves)."""
    peak_row = memory_curve.loc[memory_curve["Correlation"].idxmax()]
    peak_lag, peak_corr = int(peak_row["Lag (days)"]), peak_row["Correlation"]

    half_life = None
    tail = memory_curve[memory_curve["Lag (days)"] >= peak_lag]
    below_half = tail[tail["Correlation"] < peak_corr / 2]
    if not below_half.empty:
        half_life = float(below_half.iloc[0]["Lag (days)"])

    return peak_lag, peak_corr, half_life


def event_window(df: pd.DataFrame, event: dict, padding_days: int = 3) -> pd.DataFrame:
    start = event["start"] - pd.Timedelta(days=padding_days)
    end = event["end"] + pd.Timedelta(days=padding_days)
    window = df[(df["Time"] >= start) & (df["Time"] <= end)].copy()
    window["time_idx"] = range(len(window))
    return window
