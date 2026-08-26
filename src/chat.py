"""Together AI chat integration, with a rule-based fallback when no API key
is configured or the request fails."""

import re

import requests
import streamlit as st

from src.analysis import basin_memory_curve, memory_half_life, summarize_flood_events
from src.config import DISCHARGE_COL, PET_COL, PRECIP_COL, SOIL_MOISTURE_COL
from src.models_lstm import predict_next_step

TOGETHER_API_URL = "https://api.together.xyz/v1/completions"
COMMAND_PATTERN = re.compile(r"\{\{([^}]+)\}\}")


def get_api_key() -> str | None:
    """Read the Together AI key from Streamlit secrets or the environment.

    Never hardcode this — a prior version of this app had a live key
    committed directly in source, which is an immediate leak the moment the
    repo is public. Configure it via `.streamlit/secrets.toml` (see
    `.streamlit/secrets.toml.example`) or a TOGETHER_API_KEY env var instead.
    """
    if "api_keys" in st.secrets and "together_ai" in st.secrets["api_keys"]:
        return st.secrets["api_keys"]["together_ai"]
    import os
    return os.environ.get("TOGETHER_API_KEY")


def call_together_api(prompt: str, context: str, model: str, api_key: str) -> str:
    system_prompt = context or "You are a helpful assistant specializing in hydrology and flood prediction."
    full_prompt = f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{prompt} [/INST]"

    response = requests.post(
        TOGETHER_API_URL,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": model,
            "prompt": full_prompt,
            "max_tokens": 1024,
            "temperature": 0.3,
            "top_p": 0.7,
            "repetition_penalty": 1.0,
        },
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()

    choices = payload.get("choices")
    if not choices:
        error = payload.get("error", {}).get("message", "Unknown error")
        raise RuntimeError(f"Together AI returned no completion: {error}")
    return choices[0]["text"].strip()


def extract_analysis_commands(response: str) -> tuple[str, list[dict]]:
    """Pull {{command: params}} tags out of an LLM response."""
    commands = []
    for match in COMMAND_PATTERN.findall(response):
        parts = match.split(":")
        commands.append({
            "type": parts[0].strip(),
            "parameters": ":".join(parts[1:]).strip() if len(parts) > 1 else "",
        })
    return COMMAND_PATTERN.sub("", response).strip(), commands


def build_dataset_context(df, flood_threshold: float, model_type: str, analysis_history: list[str]) -> str:
    events = summarize_flood_events(df, flood_threshold)
    flood_days = sum(e["duration"] for e in events)

    context = f"""You are a specialized hydrologist and flood prediction assistant analyzing this dataset:

- Records: {len(df)}
- Time range: {df['Time'].min():%Y-%m-%d} to {df['Time'].max():%Y-%m-%d}
- Average discharge: {df[DISCHARGE_COL].mean():.2f} m³/s
- Maximum discharge: {df[DISCHARGE_COL].max():.2f} m³/s
- Flood threshold (95th percentile): {flood_threshold:.2f} m³/s
- Flood events: {len(events)} ({flood_days} days, {flood_days / len(df) * 100:.1f}% of record)
- Active model: {model_type.upper()}

Available columns: {', '.join(df.columns)}

Guidelines:
1. Use specific values from this dataset rather than generic statements.
2. Keep answers concise. Explain hydrological terms plainly.
3. Suggest a relevant analysis tab when it would help (time series, correlation, hysteresis, basin memory, forecasting).
4. You may embed a command like {{{{forecast: days=7}}}} to point the user at a specific view; only use: time_series, flood_events, correlation, hysteresis, basin_memory, forecast."""

    if analysis_history:
        context += "\n\nThe user has already viewed: " + ", ".join(analysis_history)

    return context


def generate_template_response(query: str, df, flood_threshold: float, model_type: str, model_bundle) -> str:
    """Rule-based fallback used when no API key is set or the API call fails."""
    q = query.lower()

    if "average discharge" in q or "mean discharge" in q:
        return f"Average discharge is {df[DISCHARGE_COL].mean():.2f} m³/s (std {df[DISCHARGE_COL].std():.2f} m³/s)."

    if "flood threshold" in q:
        return f"The flood threshold (95th percentile) is {flood_threshold:.2f} m³/s."

    if "how many flood" in q or "number of flood" in q:
        events = summarize_flood_events(df, flood_threshold)
        days = sum(e["duration"] for e in events)
        return f"{len(events)} distinct flood events, spanning {days} days ({days / len(df) * 100:.1f}% of the record)."

    if "largest flood" in q or "peak discharge" in q:
        peak_row = df.loc[df[DISCHARGE_COL].idxmax()]
        return (f"The largest event peaked at {peak_row[DISCHARGE_COL]:.2f} m³/s on "
                f"{peak_row['Time']:%Y-%m-%d} — {peak_row[DISCHARGE_COL] / flood_threshold:.1f}x the flood threshold.")

    if any(term in q for term in ("basin memory", "memory", "lag")):
        curve = basin_memory_curve(df)
        peak_lag, peak_corr, _ = memory_half_life(curve)
        return (f"Precipitation's strongest influence on discharge occurs {peak_lag} days later "
                f"(correlation {peak_corr:.3f}). See the Basin Memory tab for the full curve.")

    if "hysteresis" in q or "loop" in q:
        return ("Clockwise loops mean quick runoff response (efficient drainage, steep terrain). "
                "Counter-clockwise loops mean delayed response (storage capacity, groundwater contribution). "
                "See the Hysteresis tab to inspect specific events.")

    if any(term in q for term in ("predict", "forecast")):
        if model_type == "lstm" and model_bundle:
            discharge_pred, flood_prob, error = predict_next_step(df, model_bundle)
            if error:
                return f"Couldn't generate a forecast: {error}"
            return (f"Next-step LSTM forecast: {discharge_pred:.2f} m³/s "
                    f"(flood probability {flood_prob:.1%}). Use the Forecasting tab for a multi-day view.")
        if model_bundle:
            return "A Random Forest model is loaded — use the Forecasting tab to generate a multi-day forecast."
        return "Train or load a model from the sidebar first, then ask again."

    if "correlat" in q:
        corr_columns = [DISCHARGE_COL, PRECIP_COL, PET_COL]
        if SOIL_MOISTURE_COL in df.columns:
            corr_columns.append(SOIL_MOISTURE_COL)
        corr = df[corr_columns].corr()[DISCHARGE_COL].drop(DISCHARGE_COL)
        top = corr.abs().idxmax()
        return f"{top} correlates most strongly with discharge (r = {corr[top]:.3f}). See the Correlation tab for details."

    return (f"This dataset spans {df['Time'].min():%Y-%m-%d} to {df['Time'].max():%Y-%m-%d} "
            f"({len(df)} records). Average discharge is {df[DISCHARGE_COL].mean():.2f} m³/s, "
            f"flood threshold is {flood_threshold:.2f} m³/s. Ask about basin characteristics, "
            f"correlations, hysteresis, basin memory, or forecasts.")


def answer_query(query: str, df, flood_threshold: float, model_type: str, model_bundle,
                  analysis_history: list[str], api_key: str | None, model: str) -> tuple[str, list[dict]]:
    """Try the LLM first, fall back to templates on any failure."""
    if api_key:
        try:
            context = build_dataset_context(df, flood_threshold, model_type, analysis_history)
            response = call_together_api(query, context, model, api_key)
            return extract_analysis_commands(response)
        except Exception:
            pass  # fall through to template response

    return generate_template_response(query, df, flood_threshold, model_type, model_bundle), []
