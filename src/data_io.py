"""CSV loading/validation and export helpers."""

import base64
import io

import pandas as pd
import streamlit as st

from src.config import REQUIRED_COLUMNS


def validate_columns(df: pd.DataFrame) -> list[str]:
    """Return the required columns missing from df (empty list if valid)."""
    return [col for col in REQUIRED_COLUMNS if col not in df.columns]


@st.cache_data(show_spinner=False)
def load_dataset(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """Parse an uploaded CSV into a clean, time-sorted dataframe.

    Cached on the raw file bytes so re-running the app (e.g. after a chat
    message) doesn't re-parse and re-clean a multi-thousand-row CSV every
    time Streamlit reruns the script.
    """
    df = pd.read_csv(io.BytesIO(file_bytes))

    missing = validate_columns(df)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    df["Time"] = pd.to_datetime(df["Time"])
    df = df.replace("nan", pd.NA)
    for col in df.columns:
        if col != "Time":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.sort_values("Time").reset_index(drop=True)


def dataframe_download_link(df: pd.DataFrame, filename: str, label: str) -> str:
    """Build a base64 data-URI download link for a dataframe as CSV."""
    csv_bytes = df.to_csv(index=False).encode()
    b64 = base64.b64encode(csv_bytes).decode()
    return (
        f'<a class="download-btn" href="data:file/csv;base64,{b64}" '
        f'download="{filename}">{label}</a>'
    )


def figure_download_link(fig, filename: str, label: str) -> str:
    """Build a download link for a Plotly figure as a standalone HTML file."""
    buffer = io.StringIO()
    fig.write_html(buffer, include_plotlyjs="cdn")
    b64 = base64.b64encode(buffer.getvalue().encode()).decode()
    return (
        f'<a class="download-btn" href="data:text/html;base64,{b64}" '
        f'download="{filename}">{label}</a>'
    )
