"""Shared constants: required data schema, model paths, and Plotly styling."""

REQUIRED_COLUMNS = [
    "Time",
    "Discharge(m^3 s^-1)",
    "Precip(mm h^-1)",
    "PET(mm h^-1)",
    "SM(%)",
    "Groundwater (mm)",
    "Fast Flow(mm*1000)",
    "Slow Flow(mm*1000)",
    "Base Flow(mm*1000)",
]

DISCHARGE_COL = "Discharge(m^3 s^-1)"
PRECIP_COL = "Precip(mm h^-1)"
PET_COL = "PET(mm h^-1)"
SOIL_MOISTURE_COL = "SM(%)"

FLOOD_PERCENTILE = 0.95
SEQUENCE_LENGTH = 7
LAG_STEPS = [1, 2, 3, 7]
MAX_MEMORY_LAG_DAYS = 30

LSTM_CLASSIFICATION_PATH = "LSTM model file/g1_classification_model.h5"
LSTM_REGRESSION_PATH = "LSTM model file/g1_regression_model.h5"

TOGETHER_MODELS = [
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "meta-llama/Llama-3-8b-chat-hf",
    "meta-llama/Llama-2-70b-chat-hf",
    "mistralai/Mistral-7B-Instruct-v0.2",
]
DEFAULT_TOGETHER_MODEL = TOGETHER_MODELS[0]

# Plot styling: a single template + fixed figure size keeps every chart in the
# app visually consistent and sharp when exported, instead of each plot
# picking its own defaults.
PLOT_TEMPLATE = "plotly_white"
PLOT_HEIGHT = 560
PLOT_FONT_SIZE = 13
PLOT_EXPORT_SCALE = 3  # multiplier used when rendering PNG downloads

LEGEND_LAYOUT = dict(
    orientation="h",
    yanchor="bottom",
    y=1.02,
    xanchor="center",
    x=0.5,
)
