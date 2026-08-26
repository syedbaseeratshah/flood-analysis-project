"""Flood Peak Analysis & Prediction — Streamlit entry point.

Upload a hydro-meteorological CSV, train/load a discharge prediction model,
explore it across six analysis views, and ask questions about it in the chat
tab. See README.md for the expected CSV schema and setup.
"""

import pandas as pd
import streamlit as st

from src.analysis import (
    basin_memory_curve,
    correlation_matrix,
    event_window,
    memory_half_life,
    summarize_flood_events,
)
from src.chat import answer_query, get_api_key
from src.config import (
    DEFAULT_TOGETHER_MODEL,
    DISCHARGE_COL,
    FLOOD_PERCENTILE,
    PLOT_EXPORT_SCALE,
    TOGETHER_MODELS,
)
from src.data_io import dataframe_download_link, figure_download_link, load_dataset
from src.models_lstm import forecast_lstm, load_lstm_models
from src.models_rf import forecast_random_forest, train_random_forest
from src.plotting import (
    plot_basin_memory,
    plot_correlation_bar,
    plot_correlation_heatmap,
    plot_forecast,
    plot_hydrograph,
    plot_hysteresis_loop,
    plot_scatter_relationship,
)
from src.preprocessing import flood_threshold as compute_flood_threshold

st.set_page_config(page_title="Flood Analysis", page_icon="🌊", layout="wide")

CHART_CONFIG = {
    "toImageButtonOptions": {"format": "png", "scale": PLOT_EXPORT_SCALE},
    "displaylogo": False,
}

st.markdown("""
<style>
    .chat-message { padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem; }
    .chat-message.user { background-color: #2b313e; color: #fff; margin-left: 25%; border-radius: 0.5rem 0.5rem 0 0.5rem; }
    .chat-message.bot { background-color: #475063; color: #fff; margin-right: 25%; border-left: 4px solid #1E88E5; border-radius: 0.5rem 0.5rem 0.5rem 0; }
    .download-btn { display: inline-flex; align-items: center; padding: 0.5rem 1rem; background-color: #0d6efd;
                     color: white; border-radius: 0.5rem; text-decoration: none; margin: 0.5rem 0; font-weight: 600; }
    .download-btn:hover { background-color: #0b5ed7; color: white; }
    .model-badge { display: inline-block; padding: 0.3rem 0.6rem; font-size: 0.75rem; font-weight: 700;
                   border-radius: 0.25rem; margin-left: 0.5rem; color: white; }
    .lstm-badge { background-color: #6f42c1; }
    .rf-badge { background-color: #20c997; }
</style>
""", unsafe_allow_html=True)


def _init_session_state():
    defaults = {
        "messages": [],
        "analysis_history": [],
        "dataset": None,
        "flood_threshold": None,
        "model_type": "random_forest",
        "rf_bundle": None,
        "lstm_bundle": None,
        "llm_model": DEFAULT_TOGETHER_MODEL,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def _active_model_bundle():
    return st.session_state.lstm_bundle if st.session_state.model_type == "lstm" else st.session_state.rf_bundle


def _model_badge():
    bundle = _active_model_bundle()
    if not bundle:
        return
    if st.session_state.model_type == "lstm":
        st.markdown("<span class='model-badge lstm-badge'>LSTM Active</span>", unsafe_allow_html=True)
    else:
        st.markdown("<span class='model-badge rf-badge'>Random Forest Active</span>", unsafe_allow_html=True)


def _track_analysis(name: str):
    if name not in st.session_state.analysis_history:
        st.session_state.analysis_history.append(name)


def render_sidebar():
    with st.sidebar:
        st.header("Data & Settings")

        st.subheader("Chat Model")
        llm_model = st.selectbox("Together AI model", TOGETHER_MODELS, index=0)
        st.session_state.llm_model = llm_model
        api_key = get_api_key()
        if api_key:
            st.caption(f"API key loaded (…{api_key[-4:]})")
        else:
            st.caption("No API key configured — chat will use rule-based answers. "
                       "See .streamlit/secrets.toml.example.")

        st.markdown("---")
        uploaded_file = st.file_uploader("Upload Hydro-Meteorological CSV", type=["csv"])

        if uploaded_file is None:
            return

        try:
            df = load_dataset(uploaded_file.getvalue(), uploaded_file.name)
        except ValueError as exc:
            st.error(str(exc))
            return

        threshold = compute_flood_threshold(df, FLOOD_PERCENTILE)
        st.session_state.dataset = {"df": df, "name": uploaded_file.name}
        st.session_state.flood_threshold = threshold

        st.success(f"Loaded {uploaded_file.name}")
        st.caption(f"{len(df)} records · {df['Time'].min():%Y-%m-%d} to {df['Time'].max():%Y-%m-%d}")

        st.markdown("---")
        st.header("Prediction Model")
        model_type = st.selectbox(
            "Model type", ["random_forest", "lstm"], index=0,
            format_func=lambda x: "Random Forest" if x == "random_forest" else "LSTM (pre-trained)",
        )
        st.session_state.model_type = model_type

        if model_type == "random_forest":
            if st.button("Train Random Forest Model"):
                with st.spinner("Training..."):
                    st.session_state.rf_bundle = train_random_forest(df)
            bundle = st.session_state.rf_bundle
            if bundle:
                st.success(f"MAE {bundle['metrics']['mae']:.2f} m³/s · R² {bundle['metrics']['r2']:.2f}")
        else:
            if st.session_state.lstm_bundle is None:
                st.session_state.lstm_bundle = load_lstm_models()
            if st.session_state.lstm_bundle:
                st.success("LSTM models loaded")
            else:
                st.error("LSTM model files not found under 'LSTM model file/'.")


def render_time_series(df, threshold):
    fig = plot_hydrograph(df, "Discharge and Precipitation", threshold)
    st.plotly_chart(fig, use_container_width=True, config=CHART_CONFIG)
    col1, col2 = st.columns(2)
    col1.markdown(dataframe_download_link(df, "time_series_data.csv", "Download Data"), unsafe_allow_html=True)
    col2.markdown(figure_download_link(fig, "time_series_plot.html", "Download Plot"), unsafe_allow_html=True)

    st.subheader("Zoom to Time Window")
    col1, col2 = st.columns(2)
    start = pd.to_datetime(col1.date_input("Start Date", df["Time"].min().date()))
    end = pd.to_datetime(col2.date_input("End Date", df["Time"].max().date())) + pd.Timedelta(days=1)
    window = df[(df["Time"] >= start) & (df["Time"] < end)]

    if window.empty:
        st.warning("No data in the selected range.")
        return
    fig2 = plot_hydrograph(window, f"{start:%Y-%m-%d} to {(end - pd.Timedelta(days=1)):%Y-%m-%d}", threshold)
    st.plotly_chart(fig2, use_container_width=True, config=CHART_CONFIG)


def render_flood_events(df, threshold):
    events = summarize_flood_events(df, threshold)
    fig = plot_hydrograph(df, "Discharge with Flood Threshold", threshold)
    st.plotly_chart(fig, use_container_width=True, config=CHART_CONFIG)

    flood_days = df[DISCHARGE_COL].gt(threshold).sum()
    st.metric("Flood Days", f"{flood_days} ({flood_days / len(df) * 100:.1f}% of record)")

    if not events:
        st.info("No multi-day flood events found above the threshold.")
        return

    st.subheader(f"Top Flood Events (of {len(events)})")
    table = pd.DataFrame(events[:10])
    table["start"] = table["start"].dt.strftime("%Y-%m-%d")
    table["end"] = table["end"].dt.strftime("%Y-%m-%d")
    table["peak_date"] = table["peak_date"].dt.strftime("%Y-%m-%d")
    table = table.drop(columns=["group"]).rename(columns={
        "start": "Start", "end": "End", "duration": "Duration (days)",
        "peak": "Peak Discharge (m³/s)", "peak_date": "Peak Date", "total_precip": "Total Precip (mm)",
    })
    st.dataframe(table, use_container_width=True)
    st.markdown(dataframe_download_link(table, "flood_events.csv", "Download Flood Events"), unsafe_allow_html=True)


def render_correlation(df):
    corr = correlation_matrix(df)
    st.plotly_chart(plot_correlation_heatmap(corr), use_container_width=True, config=CHART_CONFIG)
    st.markdown(dataframe_download_link(corr.reset_index(), "correlation_matrix.csv", "Download Matrix"), unsafe_allow_html=True)

    discharge_corr = corr[DISCHARGE_COL].drop(DISCHARGE_COL).sort_values(ascending=False)
    st.plotly_chart(plot_correlation_bar(discharge_corr), use_container_width=True, config=CHART_CONFIG)

    st.subheader("Explore a Relationship")
    y_var = st.selectbox("Variable to compare against discharge", discharge_corr.index)
    sample_size = st.slider("Points to plot (sampled for readability)", 100, len(df), min(2000, len(df)))
    sample = df.sample(sample_size, random_state=0) if len(df) > sample_size else df
    st.plotly_chart(
        plot_scatter_relationship(sample, y_var, discharge_corr[y_var]),
        use_container_width=True, config=CHART_CONFIG,
    )


def render_hysteresis(df, threshold):
    events = summarize_flood_events(df, threshold, min_duration=3)
    if not events:
        st.info("No flood events long enough for hysteresis analysis.")
        return

    labels = [f"{i + 1}: {e['start']:%Y-%m-%d} to {e['end']:%Y-%m-%d} (peak {e['peak']:.2f} m³/s)"
              for i, e in enumerate(events[:5])]
    choice = st.selectbox("Flood event", labels)
    event = events[int(choice.split(":")[0]) - 1]
    window = event_window(df, event)

    col1, col2 = st.columns(2)
    fig1 = plot_hysteresis_loop(window)
    col1.plotly_chart(fig1, use_container_width=True, config=CHART_CONFIG)
    fig2 = plot_hydrograph(window, "Event Time Series", threshold)
    col2.plotly_chart(fig2, use_container_width=True, config=CHART_CONFIG)

    col1.markdown(figure_download_link(fig1, "hysteresis_loop.html", "Download Plot"), unsafe_allow_html=True)
    col2.markdown(dataframe_download_link(window, "event_data.csv", "Download Event Data"), unsafe_allow_html=True)

    st.markdown("""
**Reading the loop:** clockwise loops mean discharge responds quickly to
rainfall (efficient drainage, steep terrain); counter-clockwise loops mean a
delayed response (storage capacity, groundwater contribution). Loop width
reflects the strength of that memory effect.
    """)


def render_basin_memory(df):
    curve = basin_memory_curve(df)
    peak_lag, peak_corr, half_life = memory_half_life(curve)
    st.plotly_chart(plot_basin_memory(curve, peak_lag, half_life), use_container_width=True, config=CHART_CONFIG)

    col1, col2 = st.columns(2)
    col1.metric("Lag at Peak Correlation", f"{peak_lag} days")
    col2.metric("Peak Correlation", f"{peak_corr:.3f}")
    if half_life:
        col1.metric("Memory Half-life", f"{half_life:.0f} days")

    st.markdown(dataframe_download_link(curve, "basin_memory_data.csv", "Download Data"), unsafe_allow_html=True)
    st.markdown("""
Shorter memory (steep decline in correlation) points to efficient drainage
and a flashy response; longer memory suggests groundwater contribution and
higher storage capacity.
    """)


def render_forecasting(df, threshold):
    bundle = _active_model_bundle()
    if not bundle:
        st.warning("Train a Random Forest model or load LSTM models from the sidebar first.")
        return

    days = st.slider("Forecast horizon (days)", 1, 30, 7)
    if not st.button("Generate Forecast"):
        return

    with st.spinner("Forecasting..."):
        if st.session_state.model_type == "lstm":
            forecast = forecast_lstm(df, bundle, days)
            label = "LSTM"
        else:
            forecast = forecast_random_forest(df, bundle, days)
            label = "Random Forest"

    if forecast.empty:
        st.warning("Not enough data to forecast.")
        return

    display = forecast.copy()
    display["Date"] = display["Date"].dt.strftime("%Y-%m-%d")
    display["Discharge (m³/s)"] = display["Discharge"].round(2)
    display["Exceeds Threshold"] = (forecast["Discharge"] > threshold).map({True: "Yes", False: "No"})
    if "Flood_Probability" in forecast.columns:
        display["Flood Probability"] = (forecast["Flood_Probability"] * 100).round(1).astype(str) + "%"
        columns = ["Date", "Discharge (m³/s)", "Flood Probability", "Exceeds Threshold"]
    else:
        columns = ["Date", "Discharge (m³/s)", "Exceeds Threshold"]

    st.dataframe(display[columns], use_container_width=True)
    fig = plot_forecast(df.iloc[-30:], forecast, threshold, label)
    st.plotly_chart(fig, use_container_width=True, config=CHART_CONFIG)
    st.markdown(dataframe_download_link(display[columns], f"{label.lower()}_forecast.csv", "Download Forecast"), unsafe_allow_html=True)


ANALYSES = {
    "Time Series": render_time_series,
    "Flood Events": render_flood_events,
    "Correlation Analysis": render_correlation,
    "Hysteresis Analysis": render_hysteresis,
    "Basin Memory Analysis": render_basin_memory,
    "Forecasting": render_forecasting,
}
NO_THRESHOLD_VIEWS = {"Correlation Analysis"}


def render_dashboard_tab():
    dataset = st.session_state.dataset
    if dataset is None:
        st.info("Upload a CSV in the sidebar to begin.")
        return

    df, threshold = dataset["df"], st.session_state.flood_threshold

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Records", len(df))
    col2.metric("Time Range", f"{df['Time'].min():%Y-%m-%d} → {df['Time'].max():%Y-%m-%d}")
    col3.metric("Avg. Discharge", f"{df[DISCHARGE_COL].mean():.2f} m³/s")
    col4.metric("Flood Threshold (p95)", f"{threshold:.2f} m³/s")
    _model_badge()

    choice = st.selectbox("Analysis", list(ANALYSES.keys()))
    _track_analysis(choice)

    if choice in NO_THRESHOLD_VIEWS:
        ANALYSES[choice](df)
    else:
        ANALYSES[choice](df, threshold)


def render_chat_tab():
    dataset = st.session_state.dataset
    if dataset is None:
        st.info("Upload a CSV in the sidebar to begin.")
        return

    df, threshold = dataset["df"], st.session_state.flood_threshold
    st.caption(f"{dataset['name']} · {len(df)} records · {df['Time'].min():%Y-%m-%d} to {df['Time'].max():%Y-%m-%d}")
    _model_badge()

    for message in st.session_state.messages:
        role_class = "user" if message["role"] == "user" else "bot"
        st.markdown(f"<div class='chat-message {role_class}'>{message['content']}</div>", unsafe_allow_html=True)

    suggestions = [
        "Explain the hydrological characteristics of this basin",
        "What's the typical lag between precipitation and peak discharge?",
        "How many flood events are in this dataset?",
        "Predict discharge for the next 7 days",
    ]
    cols = st.columns(2)
    picked = None
    for i, question in enumerate(suggestions):
        if cols[i % 2].button(question, key=f"suggest_{i}"):
            picked = question

    user_input = st.chat_input("Ask a question about your data")
    query = picked or user_input

    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.spinner("Thinking..."):
            response, commands = answer_query(
                query, df, threshold, st.session_state.model_type, _active_model_bundle(),
                st.session_state.analysis_history, get_api_key(), st.session_state.llm_model,
            )
        st.session_state.messages.append({"role": "assistant", "content": response})

        for command in commands:
            label = command["type"].replace("_", " ").title()
            st.toast(f"See the '{label}' view in the Analysis Dashboard tab.")
        st.rerun()


def main():
    _init_session_state()
    render_sidebar()

    st.title("🌊 Flood Analysis and Prediction")
    tab_dashboard, tab_chat = st.tabs(["Analysis Dashboard", "Chat Assistant"])
    with tab_dashboard:
        render_dashboard_tab()
    with tab_chat:
        render_chat_tab()


if __name__ == "__main__":
    main()
