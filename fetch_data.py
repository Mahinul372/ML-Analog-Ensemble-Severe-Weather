"""
fetch_data.py — Phase 1 data pipeline (+ long-archive mode).

Fetches hourly data for Atlanta, GA (33.64, -84.42) from two Open-Meteo
sources:

* Historical *forecasts*  : Open-Meteo Previous-Runs API
  (https://previous-runs-api.open-meteo.com/v1/forecast) — the archived
  NWP forecast valid at each hour.
* Ground *observations*   : Open-Meteo Archive API (ERA5 reanalysis)
  (https://archive-api.open-meteo.com/v1/archive) — the observed state
  at each hour.

Both series are aligned on matching hourly UTC timestamps into a single
clean DataFrame with columns::

    timestamp, fcst_temp, fcst_wind, fcst_gust, fcst_press,
    obs_temp, obs_wind, obs_gust, obs_press

Units: temperature °C, wind/gust km/h, surface pressure hPa.

Modes
-----
Default (no arguments): the Phase 1 one-year archive — the past 365 days
ending 7 days ago, single request per API, saved to
``data/atlanta_weather_archive_1yr.csv``. This reproduces the 1-year file
exactly; nothing about the 1-year pipeline changed.

Long-archive mode (``--long-archive``): 2018-01-01 → 2026-11-01,
fetched in one-year chunks with a 2 s pause between chunks (to respect
Open-Meteo rate limits), saved as one file per calendar year,
``data/atlanta_weather_archive_2018.csv`` … ``_2026.csv``. The yearly
files are the canonical long-archive layout: the old combined single
file (7.15 MB) exceeded the GitHub snapshot upload size limit (HTTP 413)
and was removed; concatenating the yearly files reproduces it exactly.
The start date is NOT arbitrary: the Previous-Runs API returns all-NaN
forecasts before 2018-01-01 00:00 UTC (verified by probe, 2026-11-09),
so 2018-01-01 is the earliest date with valid forecast data. The
metrics and weights keep the ``_8yr10mo`` suffix (8 years 10 months);
the data span is 8 years 10 months (77,448 hourly rows).

Usage:
    python fetch_data.py                  # 1-year archive (default)
    python fetch_data.py --long-archive   # long archive, yearly files
    python fetch_data.py --start 2018-01-01 --end 2026-11-01 \\
        --output data/atlanta_weather_archive_2018.csv --chunk-years
"""

from __future__ import annotations

import argparse
import os
import time
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
# 1-year window ends 7 days ago to guarantee complete coverage.
N_DAYS = 365
END_DATE = date.today() - timedelta(days=7)
START_DATE = END_DATE - timedelta(days=N_DAYS - 1)

# Long-archive mode: earliest date with valid Previous-Runs forecasts
# (verified by probe 2026-11-09: all-NaN before 2018-01-01 00:00 UTC).
LONG_START_DATE = date(2018, 1, 1)
LONG_END_DATE = date(2026, 11, 1)
CHUNK_SLEEP_SECONDS = 2

DATA_DIR = Path(__file__).resolve().parent / "data"
OUTPUT_PATH = DATA_DIR / "atlanta_weather_archive_1yr.csv"

# Long-archive yearly files (canonical layout): one file per calendar
# year, 2018-01-01 -> 2026-11-01 (77,448 hourly rows total).
LONG_YEARS = list(range(2018, 2027))
LONG_YEAR_ROWS = {2018: 8760, 2019: 8760, 2020: 8784, 2021: 8760,
                  2022: 8760, 2023: 8760, 2024: 8784, 2025: 8760, 2026: 7320}


def long_year_path(year: int) -> Path:
    """Path of the long-archive file for one calendar year."""
    return DATA_DIR / f"atlanta_weather_archive_{year}.csv"


def write_yearly_files(df: pd.DataFrame) -> list[Path]:
    """Split a long-archive DataFrame by calendar year and write the
    yearly files. Verifies each file's row count; returns the paths."""
    written = []
    for year in LONG_YEARS:
        part = df[df["timestamp"].dt.year == year]
        expected = LONG_YEAR_ROWS[year]
        assert len(part) == expected, \
            f"year {year}: expected {expected} rows, got {len(part)}"
        path = long_year_path(year)
        part.to_csv(path, index=False)
        written.append(path)
        print(f"Saved {path} ({len(part)} rows)")
    return written

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
def _build_client(cache_name=".openmeteo_cache") -> openmeteo_requests.Client:
    """Cached, retrying HTTP session wrapped by the Open-Meteo client."""
    cache_session = requests_cache.CachedSession(cache_name, expire_after=-1)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    return openmeteo_requests.Client(session=retry_session)


def _fetch_hourly(client: openmeteo_requests.Client, url: str,
                  start_date: date, end_date: date) -> pd.DataFrame:
    """Fetch one hourly series from an Open-Meteo endpoint.

    Returns a DataFrame indexed by UTC timestamp with one column per
    requested variable, in the order of ``HOURLY_VARIABLES``.
    """
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
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


def _fetch_window(client, start_date, end_date):
    """Fetch + align one date window from both APIs (single request each)."""
    fcst = _fetch_hourly(client, FORECAST_URL, start_date, end_date)
    obs = _fetch_hourly(client, OBSERVATION_URL, start_date, end_date)
    merged = fcst.merge(obs, on="timestamp", suffixes=("_fcst", "_obs"))
    rename = {"timestamp": "timestamp"}
    for variable, (fcst_col, obs_col) in COLUMN_MAP.items():
        rename[f"{variable}_fcst"] = fcst_col
        rename[f"{variable}_obs"] = obs_col
    return merged.rename(columns=rename)[OUTPUT_COLUMNS]


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def build_archive(start_date=START_DATE, end_date=END_DATE,
                  chunk_years=False, cache_name=".openmeteo_cache") -> pd.DataFrame:
    """Fetch, align, clean and return the forecast/observation archive.

    With chunk_years=True the window is fetched one calendar year at a
    time with a CHUNK_SLEEP_SECONDS pause between year requests.
    """
    client = _build_client(cache_name)

    if not chunk_years:
        print(f"[{LOCATION_NAME}] Fetching ({start_date} → {end_date}) ...")
        df = _fetch_window(client, start_date, end_date)
    else:
        chunks = []
        year = start_date.year
        while year <= end_date.year:
            chunk_start = max(start_date, date(year, 1, 1))
            chunk_end = min(end_date, date(year, 12, 31))
            print(f"[{LOCATION_NAME}] Fetching year {year} "
                  f"({chunk_start} → {chunk_end}) ...")
            chunks.append(_fetch_window(client, chunk_start, chunk_end))
            if year < end_date.year:
                time.sleep(CHUNK_SLEEP_SECONDS)
            year += 1
        df = pd.concat(chunks, ignore_index=True)
        print(f"Fetched {len(chunks)} year chunks, {len(df)} rows before cleaning")

    # --- Clean: drop any row with a missing value -------------------------
    # No interpolation of any kind (reviewer revision 4): in particular
    # obs_gust is never interpolated. Verified 2026-11-09 by re-fetching
    # the full 2018-01-01 -> 2026-11-01 window with a fresh cache and no
    # interpolation: the raw APIs return 77,448 rows with 0 NaN cells in
    # every column, byte-identical to the current files — so the old
    # interpolate/ffill/bfill calls were a no-op and removing them
    # changes no value in the archive.
    n_before = len(df)
    df = df.set_index("timestamp").sort_index()
    df = df.dropna().reset_index()
    print(f"Dropped {n_before - len(df)} rows with missing values; "
          f"final rows: {len(df)}")

    return df


def verify_archive(df, expected_rows=None):
    """Check the archive is hourly-complete with no internal date gaps."""
    diffs = df["timestamp"].diff().dropna()
    n_gaps = int((diffs != pd.Timedelta(hours=1)).sum())
    print(f"Rows: {len(df)} | span: {df['timestamp'].min()} → {df['timestamp'].max()}")
    print(f"Missing values: {int(df.isna().sum().sum())} | "
          f"non-hourly gaps: {n_gaps}")
    if expected_rows is not None:
        assert len(df) == expected_rows, \
            f"expected {expected_rows} rows, got {len(df)}"
    assert n_gaps == 0, f"{n_gaps} internal date gaps found"
    assert not df.isna().any().any(), "NaN values remain in the archive"
    print("Verification passed: hourly-complete, no gaps, no NaNs.")


def main() -> Path:
    parser = argparse.ArgumentParser(description="Fetch Atlanta weather archive")
    parser.add_argument("--long-archive", action="store_true",
                        help="fetch 2018-01-01 -> 2026-11-01 in year chunks "
                             "to the yearly files data/atlanta_weather_archive_"
                             "2018.csv ... _2026.csv")
    parser.add_argument("--start", type=str, default=None,
                        help="start date YYYY-MM-DD (default: 1-year mode)")
    parser.add_argument("--end", type=str, default=None,
                        help="end date YYYY-MM-DD (default: 1-year mode)")
    parser.add_argument("--output", type=str, default=None,
                        help="output CSV path (default: data/atlanta_weather_archive_1yr.csv)")
    parser.add_argument("--chunk-years", action="store_true",
                        help="fetch one calendar year at a time (2 s pause between years)")
    args = parser.parse_args()

    if args.long_archive:
        start_date, end_date = LONG_START_DATE, LONG_END_DATE
        output_path = None  # long mode writes the yearly files, not one file
        chunk_years = True
        cache_name = ".openmeteo_cache_8yr10mo"
        expected_rows = 77448
    else:
        start_date = date.fromisoformat(args.start) if args.start else START_DATE
        end_date = date.fromisoformat(args.end) if args.end else END_DATE
        output_path = Path(args.output) if args.output else OUTPUT_PATH
        chunk_years = args.chunk_years
        cache_name = ".openmeteo_cache"
        expected_rows = None

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df = build_archive(start_date, end_date, chunk_years=chunk_years,
                       cache_name=cache_name)
    verify_archive(df, expected_rows=expected_rows)
    if args.long_archive:
        written = write_yearly_files(df)
        print(f"Saved long archive as {len(written)} yearly files "
              f"(concatenating them reproduces the 77,448-row archive)")
        return written[0].parent
    df.to_csv(output_path, index=False)
    print(f"Saved archive to {output_path}")
    print(df.head())
    print(df.describe().round(2))
    return output_path


if __name__ == "__main__":
    main()
