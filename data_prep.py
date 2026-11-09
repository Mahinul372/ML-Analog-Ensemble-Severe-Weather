"""Shared data protocol for the Analog Ensemble experiments — Phase 3.

This module is the SINGLE canonical definition of the dataset, the
chronological train/validation/test split, the feature scaler, the test
storm-event subset, and the severe-episode selection rule. Every experiment
(Traditional AnEn, Deep-AnEn-Regression, and the 3-way comparison) imports from here
so that preprocessing and splits are never recreated independently.

Rules enforced here:
  * The feature scaler is fitted on the TRAINING split only. No validation
    or test information enters the scaler.
  * Splits are strictly chronological (no shuffling).
  * The evaluation subset is defined ONCE per regime: the test-window
    timestamps with observed gust >= 35 km/h. All methods evaluate on
    those exact timestamps.
  * The plotted severe episode is selected by a PREDEFINED rule (the
    contiguous run of storm hours with the highest observed peak gust),
    not by visual inspection of which event flatters a model.

Two regimes share this protocol, branched by which CSV file is passed
(see regime_for_path):
  * "1yr"  — data/atlanta_weather_archive.csv (2025-11-02 → 2026-11-01):
    train ≤ 2026-08-15 (6,888 rows), val → 2026-08-31 (384), test
    ≥ 2026-09-01 (1,488 rows, 130 storm events).
  * "long" — data/atlanta_weather_archive_10yr.csv (2018-01-01 →
    2026-11-01, 77,448 rows): train 2018-01-01 → 2024-11-01 (59,928
    rows), val 2024-11-02 → 2025-11-01 (8,760), test 2025-11-02 →
    2026-11-01 (8,760 rows). The archive starts 2018-01-01 because the
    Open-Meteo Previous-Runs API returns no valid forecasts earlier
    (verified by probe 2026-11-09). The test window is IDENTICAL in
    calendar terms to the 1-year file's full span, but the training pool
    is ~8.7x larger — that is the scale-up comparison.
"""

import os

import pandas as pd
from sklearn.preprocessing import StandardScaler

FEATURES = ["fcst_temp", "fcst_wind", "fcst_gust", "fcst_press"]
TARGET = "obs_gust"

# Chronological splits (UTC). Identical to the Phase 2 Deep-AnEn-Regression experiment.
TRAIN_END = pd.Timestamp("2026-08-15 23:00", tz="UTC")
VAL_END = pd.Timestamp("2026-08-31 23:00", tz="UTC")

# Long-archive splits (UTC): train 2018-01-01 -> 2024-11-01 (59,928 rows),
# val 2024-11-02 -> 2025-11-01 (8,760), test 2025-11-02 -> 2026-11-01 (8,760).
LONG_TRAIN_END = pd.Timestamp("2024-11-01 23:00", tz="UTC")
LONG_VAL_END = pd.Timestamp("2025-11-01 23:00", tz="UTC")

# Regime split boundaries, keyed by regime name.
SPLIT_BOUNDARIES = {
    "1yr": (TRAIN_END, VAL_END),
    "long": (LONG_TRAIN_END, LONG_VAL_END),
}

EXCLUSION_HOURS = 7 * 24  # +/-7-day window around each target timestamp
K = 15                    # ensemble size for both AnEn variants
STORM_THRESHOLD = 35.0    # km/h; defines the test storm-event subset


def regime_for_path(path):
    """Which split regime a CSV file belongs to, by filename.

    Files with "10yr" in the name use the long-archive splits; everything
    else uses the 1-year splits. The 1-year file is the default, so any
    caller that never passes a path keeps the original behaviour.
    """
    name = os.path.basename(str(path)) if path is not None else ""
    return "long" if "10yr" in name else "1yr"


def load_archive(path=None):
    """Load data/atlanta_weather_archive.csv, sorted by timestamp."""
    if path is None:
        here = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(here, "data", "atlanta_weather_archive.csv")
    df = pd.read_csv(path, parse_dates=["timestamp"])
    return df.sort_values("timestamp").reset_index(drop=True)


def split_archive(df, regime="1yr"):
    """Chronological train / validation / test splits (no shuffling).

    regime="1yr" (default) keeps the Phase 2/3/4 splits exactly;
    regime="long" uses the long-archive hard chronological barrier.
    """
    if regime not in SPLIT_BOUNDARIES:
        raise ValueError(f"unknown regime {regime!r} (expected '1yr' or 'long')")
    train_end, val_end = SPLIT_BOUNDARIES[regime]
    train_df = df[df["timestamp"] <= train_end].reset_index(drop=True)
    val_df = df[
        (df["timestamp"] > train_end) & (df["timestamp"] <= val_end)
    ].reset_index(drop=True)
    test_df = df[df["timestamp"] > val_end].reset_index(drop=True)
    return train_df, val_df, test_df


def load_and_split(path=None, regime=None):
    """Load an archive and split it under the regime for that file.

    The regime is detected from the filename unless given explicitly.
    Returns (df, train_df, val_df, test_df, regime).
    """
    df = load_archive(path)
    if regime is None:
        regime = regime_for_path(path)
    train_df, val_df, test_df = split_archive(df, regime=regime)
    return df, train_df, val_df, test_df, regime


def fit_feature_scaler(train_df):
    """StandardScaler fitted on the TRAINING split only."""
    return StandardScaler().fit(train_df[FEATURES])


def get_storm_events(test_df, threshold=STORM_THRESHOLD):
    """The explicit evaluation subset: test timestamps with obs_gust >= 35.

    Every method evaluates on these exact timestamps — the list is built
    once here and passed to each method, never re-derived per method.
    """
    events = test_df[test_df[TARGET] >= threshold].reset_index(drop=True)
    return events


def get_severe_episode(test_df, threshold=STORM_THRESHOLD, pad_hours=24):
    """Select the plotted severe episode by a predefined rule.

    Rule: among all maximal contiguous runs of test-window hours with
    observed gust >= threshold, take the run with the highest observed
    peak gust (ties broken by longest run). The plotted window is that
    episode padded by `pad_hours` on each side.

    Returns a dict with episode start/end, peak time/gust, and window
    start/end. For this archive: episode 2026-10-10 01:00–13:00 UTC
    (13 h, peak 71.6 km/h), window 2026-10-09 01:00 → 2026-10-11 13:00.
    """
    storm = (test_df[TARGET] >= threshold).values
    runs = []
    i = 0
    while i < len(test_df):
        if storm[i]:
            j = i
            while j + 1 < len(test_df) and storm[j + 1]:
                j += 1
            runs.append((i, j))
            i = j + 1
        else:
            i += 1
    if not runs:
        raise ValueError("No storm episode found in the test window.")
    # Highest peak first, longest run as tie-break.
    best = max(
        runs,
        key=lambda r: (
            test_df.loc[r[0]:r[1], TARGET].max(),
            r[1] - r[0],
        ),
    )
    start = test_df.loc[best[0], "timestamp"]
    end = test_df.loc[best[1], "timestamp"]
    peak_idx = test_df.loc[best[0]:best[1], TARGET].idxmax()
    return {
        "episode_start": start,
        "episode_end": end,
        "peak_timestamp": test_df.loc[peak_idx, "timestamp"],
        "peak_gust": float(test_df.loc[peak_idx, TARGET]),
        "window_start": start - pd.Timedelta(hours=pad_hours),
        "window_end": end + pd.Timedelta(hours=pad_hours),
        "n_runs": len(runs),
    }
