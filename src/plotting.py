"""Plotly figure builders. One shared layout function keeps every chart in
the app at the same resolution, font size, and legend placement instead of
each analysis tab hand-rolling its own (which is how the original app drifted
into inconsistent, low-res, overlapping-legend charts)."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.config import DISCHARGE_COL, LEGEND_LAYOUT, PLOT_HEIGHT, PLOT_TEMPLATE, PRECIP_COL, SOIL_MOISTURE_COL


def _apply_hd_layout(fig: go.Figure, title: str, **extra_layout) -> go.Figure:
    fig.update_layout(
        title=title,
        template=PLOT_TEMPLATE,
        height=PLOT_HEIGHT,
        legend=LEGEND_LAYOUT,
        margin=dict(t=70, b=40, l=60, r=60),
        **extra_layout,
    )
    return fig


def plot_hydrograph(df: pd.DataFrame, title: str, threshold: float | None = None) -> go.Figure:
    """Discharge + precipitation (+ soil moisture) overlay. Used for both the
    full-record view and the zoomed date-range view, so it only needs to
    exist once."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Time"], y=df[DISCHARGE_COL], name="Discharge (m³/s)"))
    fig.add_trace(go.Bar(x=df["Time"], y=df[PRECIP_COL], name="Precipitation (mm/h)", opacity=0.45, yaxis="y2"))

    if SOIL_MOISTURE_COL in df.columns:
        fig.add_trace(go.Scatter(
            x=df["Time"], y=df[SOIL_MOISTURE_COL], name="Soil Moisture (%)",
            yaxis="y3", line=dict(dash="dash"),
        ))

    if threshold is not None:
        fig.add_trace(go.Scatter(
            x=df["Time"], y=[threshold] * len(df), name="Flood Threshold",
            line=dict(dash="dash", color="crimson"),
        ))

    return _apply_hd_layout(
        fig, title,
        yaxis_title="Discharge (m³/s)",
        yaxis2=dict(title="Precipitation (mm/h)", overlaying="y", side="right"),
        yaxis3=dict(title="Soil Moisture (%)", overlaying="y", side="right", position=0.97, showgrid=False),
    )


def plot_correlation_heatmap(corr: pd.DataFrame) -> go.Figure:
    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
    return _apply_hd_layout(fig, "Correlation Matrix")


def plot_correlation_bar(discharge_corr: pd.Series) -> go.Figure:
    fig = px.bar(
        x=discharge_corr.index, y=discharge_corr.values,
        labels={"x": "Variable", "y": "Correlation Coefficient"},
        color=discharge_corr.values, color_continuous_scale="RdBu_r", range_color=[-1, 1],
    )
    fig.update_layout(showlegend=False)
    return _apply_hd_layout(fig, "Variables Correlated with Discharge")


def plot_scatter_relationship(df: pd.DataFrame, y_var: str, correlation: float) -> go.Figure:
    fig = px.scatter(df, x=DISCHARGE_COL, y=y_var, trendline="ols", opacity=0.55)
    return _apply_hd_layout(fig, f"Discharge vs {y_var} (r = {correlation:.3f})")


def plot_hysteresis_loop(event_data: pd.DataFrame) -> go.Figure:
    """Precipitation-vs-discharge loop, colored by time progression.

    Direction is conveyed with a colorscale + a handful of arrow markers
    rather than one annotation per data point (the original implementation
    added an arrow for every single row, which got slow and visually
    cluttered on longer flood events).
    """
    fig = px.scatter(
        event_data, x=PRECIP_COL, y=DISCHARGE_COL,
        color="time_idx", color_continuous_scale="Viridis",
        labels={"time_idx": "Time Step"},
    )
    fig.update_traces(mode="lines+markers", line=dict(color="rgba(120,120,120,0.35)"))

    arrow_every = max(1, len(event_data) // 6)
    for i in range(0, len(event_data) - 1, arrow_every):
        fig.add_annotation(
            x=event_data[PRECIP_COL].iloc[i + 1], y=event_data[DISCHARGE_COL].iloc[i + 1],
            ax=event_data[PRECIP_COL].iloc[i], ay=event_data[DISCHARGE_COL].iloc[i],
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1.5, arrowcolor="crimson",
        )

    fig.update_layout(coloraxis_colorbar=dict(title="Time Step"))
    return _apply_hd_layout(fig, "Hysteresis Loop (Precipitation vs Discharge)", showlegend=False)


def plot_basin_memory(memory_curve: pd.DataFrame, peak_lag: int, half_life: float | None) -> go.Figure:
    fig = px.line(memory_curve, x="Lag (days)", y="Correlation", markers=True)
    fig.add_vline(
        x=peak_lag, line_dash="dash", line_color="crimson",
        annotation_text=f"Peak correlation at {peak_lag}d", annotation_position="top right",
    )
    if half_life is not None:
        fig.add_vline(
            x=half_life, line_dash="dot", line_color="darkorange",
            annotation_text=f"Half-life at {half_life:.0f}d", annotation_position="bottom right",
        )
    return _apply_hd_layout(fig, "Basin Memory: Discharge vs Lagged Precipitation", showlegend=False)


def plot_forecast(history: pd.DataFrame, forecast: pd.DataFrame, threshold: float, model_label: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=history["Time"], y=history[DISCHARGE_COL], name="Historical Discharge", line=dict(color="#1E88E5")))
    fig.add_trace(go.Scatter(x=forecast["Date"], y=forecast["Discharge"], name=f"{model_label} Forecast", line=dict(color="#E64A19")))

    if "Flood_Probability" in forecast.columns:
        fig.add_trace(go.Bar(
            x=forecast["Date"], y=forecast["Flood_Probability"], name="Flood Probability",
            marker_color="rgba(128,0,128,0.35)", yaxis="y2",
        ))

    timeline = pd.concat([history["Time"], forecast["Date"]])
    fig.add_trace(go.Scatter(
        x=timeline, y=[threshold] * len(timeline), name="Flood Threshold",
        line=dict(dash="dash", color="crimson"),
    ))

    return _apply_hd_layout(
        fig, f"{len(forecast)}-Day Discharge Forecast ({model_label})",
        xaxis_title="Date", yaxis_title="Discharge (m³/s)",
        yaxis2=dict(title="Flood Probability", overlaying="y", side="right", range=[0, 1], showgrid=False),
    )
