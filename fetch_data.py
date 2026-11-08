"""
fetch_data.py — Phase 1 data pipeline.

Fetches hourly data for Atlanta, GA (33.64, -84.42) spanning the past
365 days from two Open-Meteo sources:

* Historical *forecasts*  : Open-Meteo Previous-Runs API
  (https://previous-runs-api.open-meteo.com/v1/forecast) — the archived
  NWP forecast valid at each hour.
* Ground *observations*   : Open-Meteo Archive API (ERA5 reanalysis)
  (https://archive-api.open-meteo.com/v1/archive) — the observed state
  at each hour.

Both series are aligned on matching hourly UTC timestamps into a single
clean DataFrame and saved to ``data/atlanta_weather_archive.csv`` with
columns::

    timestamp, fcst_temp, fcst_wind, fcst_gust, fcst_press,
    obs_temp, obs_wind, obs_gust, obs_press

Units: temperature °C, wind/gust km/h, surface pressure hPa.
"""

from __future__ import annotations

import os
from datetime import date, timedelta
from pathlib import Path

import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
LATITUDE = 33.64
LONGITUDE = -84.42
LOCATION_NAME = "Atlanta, GA"

FORECAST_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
OBSERVATION_URL = "https://archive-api.open-meteo.com/v1/archive"

HOURLY_VARIABLES = [
    "temperature_2m",
    "wind_speed_10m",
    "wind_gusts_10m",
    "surface_pressure",
]

# ERA5 reanalysis in the Archive API lags real time by ~5 days, so the
# window ends 7 days ago to guarantee complete coverage.
N_DAYS = 365
END_DATE = date.today() - timedelta(days=7)
START_DATE = END_DATE - timedelta(days=N_DAYS - 1)

DATA_DIR = Path(__file__).resolve().parent / "data"
OUTPUT_PATH = DATA_DIR / "atlanta_weather_archive.csv"

# Explicit output column names: fcst_* = NWP forecast, obs_* = observation.
COLUMN_MAP = {
    "temperature_2m": ("fcst_temp", "obs_temp"),
    "wind_speed_10m": ("fcst_wind", "obs_wind"),
    "wind_gusts_10m": ("fcst_gust", "obs_gust"),
    "surface_pressure": ("fcst_press", "obs_press"),
}
OUTPUT_COLUMNS = [
    "timestamp",
    "fcst_temp", "fcst_wind", "fcst_gust", "fcst_press",
    "obs_temp", "obs_wind", "obs_gust", "obs_press",
]


# ---------------------------------------------------------------------------
# Open-Meteo client
# ---------------------------------------------------------------------------
def _build_client() -> openmeteo_requests.Client:
    """Cached, retrying HTTP session wrapped by the Open-Meteo client."""
    cache_session = requests_cache.CachedSession(".openmeteo_cache", expire_after=-1)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    return openmeteo_requests.Client(session=retry_session)


def _fetch_hourly(client: openmeteo_requests.Client, url: str) -> pd.DataFrame:
    """Fetch one hourly series from an Open-Meteo endpoint.

    Returns a DataFrame indexed by UTC timestamp with one column per
    requested variable, in the order of ``HOURLY_VARIABLES``.
    """
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "start_date": START_DATE.isoformat(),
        "end_date": END_DATE.isoformat(),
        "hourly": HOURLY_VARIABLES,
        "timezone": "UTC",
    }
    responses = client.weather_api(url, params=params)
    response = responses[0]

    hourly = response.Hourly()
    times = pd.date_range(
        start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
        end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive="left",
    )

    data = {"timestamp": times}
    for i, variable in enumerate(HOURLY_VARIABLES):
        data[variable] = hourly.Variables(i).ValuesAsNumpy()

    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def build_archive() -> pd.DataFrame:
    """Fetch, align, clean and return the forecast/observation archive."""
    client = _build_client()

    print(f"[{LOCATION_NAME}] Fetching NWP forecasts "
          f"({START_DATE} → {END_DATE}) ...")
    fcst = _fetch_hourly(client, FORECAST_URL)

    print(f"[{LOCATION_NAME}] Fetching ERA5 observations "
          f"({START_DATE} → {END_DATE}) ...")
    obs = _fetch_hourly(client, OBSERVATION_URL)

    # --- Align on matching hourly timestamps (inner join) ------------------
    merged = fcst.merge(obs, on="timestamp", suffixes=("_fcst", "_obs"))
    print(f"Aligned rows before cleaning: {len(merged)}")

    # --- Rename to explicit fcst_*/obs_* columns ---------------------------
    rename = {"timestamp": "timestamp"}
    for variable, (fcst_col, obs_col) in COLUMN_MAP.items():
        rename[f"{variable}_fcst"] = fcst_col
        rename[f"{variable}_obs"] = obs_col
    df = merged.rename(columns=rename)[OUTPUT_COLUMNS]

    # --- Clean: interpolate minor gaps, then drop any remaining NaNs --------
    n_before = len(df)
    df = df.set_index("timestamp").sort_index()
    df = df.interpolate(method="time", limit=3)   # short gaps only
    df = df.ffill(limit=3).bfill(limit=3)
    df = df.dropna().reset_index()
    print(f"Dropped {n_before - len(df)} rows with unrecoverable gaps; "
          f"final rows: {len(df)}")

    return df


def main() -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df = build_archive()
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved archive to {OUTPUT_PATH}")
    print(df.head())
    print(df.describe().round(2))
    return OUTPUT_PATH


if __name__ == "__main__":
    main()
