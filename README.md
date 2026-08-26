# Flood Peak Analysis & Prediction

A Streamlit dashboard for exploring hydro-meteorological time series and
forecasting river discharge. It works with catchment datasets containing
discharge, precipitation, soil moisture, groundwater, and flow-component
series. Two prediction paths are supported: a Random Forest trained on the
fly, and a pair of pre-trained LSTM models (classification for flood
probability, regression for discharge).

## What it does

- Six analysis views: time series, flood event detection, correlation
  analysis, hysteresis loops, basin memory (lagged precipitation response),
  and multi-day forecasting.
- Two interchangeable prediction models. Train a Random Forest per session,
  or load the bundled LSTM models.
- A chat tab that answers questions about the loaded dataset. It uses
  Together AI when a key is configured, and falls back to rule-based answers
  otherwise. The app works with zero extra setup beyond `pip install`.
- CSV and HTML exports for every table and chart.

## Results

Metrics below come from `scripts/run_analysis.py`, run against the three
sample catchments (`g1`, `g2`, `g3`), each with 12,197 daily records from
1991 to 2024. Random Forest is trained fresh on each dataset with an 80/20
time-based split.

| Catchment | Flood threshold (p95) | Flood events | RF MAE | RF R² |
|---|---|---|---|---|
| g1 | 173.53 m3/s | 138 | 1.62 m3/s | 0.997 |
| g2 | 86.50 m3/s | 139 | 0.94 m3/s | 0.997 |
| g3 | 0.92 m3/s | 130 | 0.005 m3/s | 0.999 |

The high R² scores reflect that discharge is strongly autocorrelated day to
day. Lag-1 discharge is one of the model's input features, so the model is
mostly learning "tomorrow looks like today plus a precipitation-driven
adjustment." This is expected and reasonable for a single-step forecast, but
it means R² alone should not be read as a measure of how well the model
predicts flood peaks specifically. The multi-day forecast in the app
degrades further out for the same reason, since each step depends on the
model's own earlier prediction.

Basin memory analysis also shows a consistent pattern across all three
catchments: peak correlation between precipitation and discharge occurs at
a 1-day lag, with the correlation halving by around day 3. This lines up
with a fast-draining catchment rather than one with strong groundwater
storage.

Basin memory computation runs in under 50ms per catchment after switching
from a per-lag Python loop to a single vectorized `pd.concat` and
`corrwith` call. Data loading and feature engineering are cached with
`st.cache_data`, so repeated Streamlit reruns (which happen on every chat
message and every widget interaction) do not reprocess the full dataset
each time.

## Project layout

```
app.py                          # Streamlit entry point / page layout
src/
  config.py                     # schema, model paths, plot styling constants
  data_io.py                    # CSV loading/validation, download links
  preprocessing.py              # shared feature engineering (RF + LSTM)
  models_rf.py                  # Random Forest training and forecasting
  models_lstm.py                # LSTM loading, single-step and multi-day forecasting
  analysis.py                   # flood events, correlation, basin memory (data only)
  plotting.py                   # Plotly figure builders
  chat.py                       # Together AI call + template fallback
scripts/
  run_analysis.py               # command-line check, no Streamlit required
LSTM model file/                # pre-trained g1 classification/regression models
sample.ipynb                    # exploratory notebook the models were developed in
data/                           # your CSVs go here, gitignored, not part of the repo
```

## Data format

Each CSV needs these columns with exact names and a header row:

```
Time, Discharge(m^3 s^-1), Precip(mm h^-1), PET(mm h^-1), SM(%),
Groundwater (mm), Fast Flow(mm*1000), Slow Flow(mm*1000), Base Flow(mm*1000)
```

`Time` should parse as a timestamp. `YYYY-MM-DD HH:MM` works. Missing
values can be blank or the literal string `nan`.

Data files are not committed to this repo. `.gitignore` excludes `data/`
and `*.csv`. Put your own catchment CSVs in `data/` locally, or upload one
directly through the app's sidebar.

## Setup

```bash
git clone https://github.com/syedbaseeratshah/flood-analysis-project.git
cd flood-analysis-project
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Chat is optional. To enable Together AI instead of the rule-based fallback,
copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and add
your key. Do not commit this file. It is gitignored.

```toml
[api_keys]
together_ai = "your_together_ai_api_key_here"
```

To check the pipeline without launching the UI:

```bash
python scripts/run_analysis.py data/ts.g1.crestphys.csv
```

Then run the full app:

```bash
streamlit run app.py
```

## Using it

1. Upload a CSV from the sidebar.
2. Pick a model type. Random Forest trains on your data. LSTM loads the
   pre-trained `g1` models.
3. Step through the analysis views in the Analysis Dashboard tab.
4. Ask questions in the Chat Assistant tab, or click one of the suggested
   prompts.

## Notes on the forecasting

Both forecast paths roll a model forward day by day, using its own previous
prediction as the next input. Real future precipitation is not available,
so this is a standard simplification for demo and exploratory forecasting.
It is not a substitute for a model that ingests an actual weather forecast.
Accuracy degrades the further out you forecast, particularly for
storm-driven discharge spikes.
