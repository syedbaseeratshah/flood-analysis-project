# Flood Peak Analysis & Prediction

A Streamlit dashboard for exploring hydro-meteorological time series and
forecasting river discharge, built around a set of catchment datasets
(discharge, precipitation, soil moisture, groundwater, and flow-component
series) and two prediction paths: a Random Forest trained on the fly, and a
pair of pre-trained LSTM models (classification for flood probability,
regression for discharge).

## What it does

- Six analysis views: time series, flood event detection, correlation
  analysis, hysteresis loops, basin memory (lagged precipitation response),
  and multi-day forecasting.
- Two interchangeable prediction models — train a Random Forest per session,
  or load the bundled LSTM models.
- A chat tab that answers questions about the loaded dataset. It uses
  Together AI when a key is configured, and falls back to rule-based answers
  otherwise, so the app works with zero setup beyond `pip install`.
- CSV and HTML exports for every table and chart.

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
LSTM model file/                # pre-trained g1 classification/regression models
sample.ipynb                    # exploratory notebook the models were developed in
data/                           # your CSVs go here — gitignored, not part of the repo
```

## Data format

Each CSV needs these columns (exact names, header row required):

```
Time, Discharge(m^3 s^-1), Precip(mm h^-1), PET(mm h^-1), SM(%),
Groundwater (mm), Fast Flow(mm*1000), Slow Flow(mm*1000), Base Flow(mm*1000)
```

`Time` should parse as a timestamp (`YYYY-MM-DD HH:MM` works). Missing
values can be blank or the literal string `nan`.

Data files are not committed to this repo — `.gitignore` excludes `data/`
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
your key (never commit this file — it's gitignored):

```toml
[api_keys]
together_ai = "your_together_ai_api_key_here"
```

Then run:

```bash
streamlit run app.py
```

## Using it

1. Upload a CSV from the sidebar.
2. Pick a model type (Random Forest trains on your data; LSTM loads the
   pre-trained `g1` models) and train/load it.
3. Step through the analysis views in the Analysis Dashboard tab.
4. Ask questions in the Chat Assistant tab, or click one of the suggested
   prompts.

## Notes on the forecasting

Both forecast paths roll a model forward day by day using its own previous
prediction as the next input, since no real future precipitation is
available. This is a standard simplification for demo/exploratory
forecasting, not a substitute for a model that ingests an actual weather
forecast — accuracy degrades the further out you forecast, particularly for
storm-driven discharge spikes.
