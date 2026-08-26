"""Standalone check: run the pipeline without Streamlit and print metrics.

Trains the Random Forest, prints flood/correlation/basin-memory stats, and
saves a couple of the HD charts as standalone HTML files you can open
directly in a browser. Useful for a first pass before loading the full app.

Usage:
    python scripts/run_analysis.py data/ts.g1.crestphys.csv
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analysis import basin_memory_curve, correlation_matrix, memory_half_life, summarize_flood_events
from src.data_io import load_dataset
from src.models_rf import forecast_random_forest, train_random_forest
from src.plotting import plot_basin_memory, plot_correlation_heatmap, plot_hydrograph
from src.preprocessing import flood_threshold


def main(csv_path: str):
    with open(csv_path, "rb") as f:
        raw = f.read()

    df = load_dataset(raw, Path(csv_path).name)
    print(f"Loaded {len(df)} records: {df['Time'].min():%Y-%m-%d} to {df['Time'].max():%Y-%m-%d}")

    threshold = flood_threshold(df, 0.95)
    events = summarize_flood_events(df, threshold)
    print(f"Flood threshold (p95): {threshold:.2f} m3/s")
    print(f"Flood events: {len(events)}, total days: {sum(e['duration'] for e in events)}")

    t0 = time.perf_counter()
    memory_curve = basin_memory_curve(df, max_lag=30)
    peak_lag, peak_corr, half_life = memory_half_life(memory_curve)
    print(f"Basin memory computed in {time.perf_counter() - t0:.3f}s "
          f"-> peak lag {peak_lag}d (r={peak_corr:.3f}), half-life {half_life}d")

    print("\nTraining Random Forest...")
    t0 = time.perf_counter()
    rf_bundle = train_random_forest(df)
    print(f"Trained in {time.perf_counter() - t0:.2f}s")
    print(f"MAE: {rf_bundle['metrics']['mae']:.3f} m3/s, R2: {rf_bundle['metrics']['r2']:.4f}")

    forecast = forecast_random_forest(df, rf_bundle, days=7)
    print("\n7-day forecast:")
    print(forecast.to_string(index=False))

    out_dir = Path("scripts/output")
    out_dir.mkdir(exist_ok=True)

    corr = correlation_matrix(df)
    plot_hydrograph(df.tail(2000), "Discharge and Precipitation (last 2000 records)", threshold).write_html(
        out_dir / "hydrograph.html", include_plotlyjs="cdn"
    )
    plot_correlation_heatmap(corr).write_html(out_dir / "correlation.html", include_plotlyjs="cdn")
    plot_basin_memory(memory_curve, peak_lag, half_life).write_html(out_dir / "basin_memory.html", include_plotlyjs="cdn")
    print(f"\nSaved charts to {out_dir}/ — open the .html files directly in a browser.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/run_analysis.py <path-to-csv>")
        sys.exit(1)
    main(sys.argv[1])
