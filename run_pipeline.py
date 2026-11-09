"""
run_pipeline.py — Phase 1 end-to-end execution.

1. Runs the data pipeline (``fetch_data.py``) to generate
   ``data/atlanta_weather_archive_1yr.csv``.
2. Instantiates ``AnalogEnsembleEngine`` on the archive.
3. Runs a test query on a sample severe-weather day: the timestamp of
   the peak observed wind gust in the dataset.
4. Prints raw forecast vs. actual observation vs. AnEn ensemble metrics
   (mean, 10th and 90th percentiles) to verify the code end-to-end.

Usage:
    python run_pipeline.py [--skip-fetch]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analog_ensemble import AnalogEnsembleEngine  # noqa: E402
from fetch_data import OUTPUT_PATH, main as fetch_main  # noqa: E402


def _fmt_row(label: str, values: dict) -> str:
    return (
        f"{label:<14}"
        f"temp {values['obs_temp']:6.2f} °C  "
        f"wind {values['obs_wind']:6.2f} km/h  "
        f"gust {values['obs_gust']:6.2f} km/h  "
        f"press {values['obs_press']:7.2f} hPa"
    )


def _fmt_fcst_row(label: str, values: dict) -> str:
    return (
        f"{label:<14}"
        f"temp {values['fcst_temp']:6.2f} °C  "
        f"wind {values['fcst_wind']:6.2f} km/h  "
        f"gust {values['fcst_gust']:6.2f} km/h  "
        f"press {values['fcst_press']:7.2f} hPa"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1 pipeline runner")
    parser.add_argument(
        "--skip-fetch",
        action="store_true",
        help="Reuse the existing CSV instead of re-fetching from Open-Meteo.",
    )
    args = parser.parse_args()

    # 1. Data pipeline ------------------------------------------------------
    if args.skip_fetch and OUTPUT_PATH.exists():
        print(f"Reusing existing archive: {OUTPUT_PATH}")
    else:
        fetch_main()

    # 2. Load archive and build the engine ----------------------------------
    df = pd.read_csv(OUTPUT_PATH, parse_dates=["timestamp"])
    print(f"\nArchive loaded: {len(df)} hourly rows, "
          f"{df['timestamp'].min()} → {df['timestamp'].max()}")
    engine = AnalogEnsembleEngine(df)

    # 3. Test query: peak observed wind-gust timestamp (severe weather) -----
    peak_idx = df["obs_gust"].idxmax()
    target = df.loc[peak_idx, "timestamp"]
    print(f"\nSevere-weather test case: peak observed gust "
          f"{df.loc[peak_idx, 'obs_gust']:.1f} km/h at {target}")

    result = engine.find_analogs(target, K=10)

    # 4. Report --------------------------------------------------------------
    print("\n" + "=" * 78)
    print(f"ANALOG ENSEMBLE — target {result['target_timestamp']} (K=10)")
    print("=" * 78)
    print(_fmt_fcst_row("Forecast:", result["target_forecast"]))
    print(_fmt_row("Observed:", result["target_observation"]))
    print("-" * 78)
    print("Ensemble (observations at the 10 analog times):")
    print(_fmt_row("Mean:", result["ensemble_mean"]))
    print(_fmt_row("Median:", result["ensemble_median"]))
    print(_fmt_row("10th pct:", result["p10"]))
    print(_fmt_row("90th pct:", result["p90"]))
    print("-" * 78)
    print("Top 10 analog timestamps (closest first):")
    for i, analog in enumerate(result["analogs"], start=1):
        print(f"  {i:2d}. {analog['timestamp']}  distance={analog['distance']:.3f}")
    print("=" * 78)

    # Quick sanity check: the ensemble spread should bracket the observation
    # more often than the raw forecast alone in a well-calibrated setup.
    obs_gust = result["target_observation"]["obs_gust"]
    p10_gust = result["p10"]["obs_gust"]
    p90_gust = result["p90"]["obs_gust"]
    inside = p10_gust <= obs_gust <= p90_gust
    print(f"\nObserved gust {obs_gust:.1f} km/h inside AnEn 10–90% interval "
          f"[{p10_gust:.1f}, {p90_gust:.1f}] km/h: {inside}")
    print("\nPhase 1 pipeline completed successfully.")


if __name__ == "__main__":
    main()
