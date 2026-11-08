"""
analog_ensemble.py — Analog Ensemble (AnEn) search engine.

Phase 1 core: given a target timestamp, the engine finds the K historical
timestamps whose *forecast* feature vectors most closely resemble the
forecast at the target time, then builds an ensemble from the *observed*
weather at those analog times. This is the standard NWP post-processing
formulation (Delle Monache et al., 2011, 2013): similar forecasts imply
similar forecast errors, so the distribution of past observations at
analog times calibrates the current forecast.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

FORECAST_FEATURES = ["fcst_temp", "fcst_wind", "fcst_gust", "fcst_press"]
OBSERVED_VARIABLES = ["obs_temp", "obs_wind", "obs_gust", "obs_press"]

# Hours excluded on either side of the target timestamp to prevent the
# target's own weather (and its immediate neighbours) leaking into the
# analog search.
EXCLUSION_WINDOW_HOURS = 7 * 24


@dataclass
class AnalogEnsembleEngine:
    """Analog Ensemble search over a forecast/observation archive.

    Parameters
    ----------
    df:
        Archive DataFrame as produced by ``fetch_data.py`` — columns
        ``timestamp`` plus ``fcst_*`` forecast features and ``obs_*``
        observations. Must be sorted by timestamp (sorted on init).
    """

    df: pd.DataFrame
    _scaler: StandardScaler = field(init=False, repr=False)
    _scaled_features: np.ndarray = field(init=False, repr=False)
    _timestamps: pd.Series = field(init=False, repr=False)

    def __post_init__(self) -> None:
        df = self.df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.sort_values("timestamp").reset_index(drop=True)
        missing = set(FORECAST_FEATURES + OBSERVED_VARIABLES) - set(df.columns)
        if missing:
            raise ValueError(f"Archive is missing columns: {sorted(missing)}")
        self.df = df
        self._timestamps = df["timestamp"]

        # Standardize forecast features once, over the full archive, so
        # distances are comparable across variables with different units.
        self._scaler = StandardScaler()
        self._scaled_features = self._scaler.fit_transform(
            df[FORECAST_FEATURES].to_numpy(dtype=float)
        )

    # ------------------------------------------------------------------
    def _target_row(self, target_timestamp) -> tuple[int, np.ndarray]:
        """Locate the target row and return (index, scaled feature vector)."""
        target = pd.to_datetime(target_timestamp, utc=True)
        matches = np.where(self._timestamps == target)[0]
        if len(matches) == 0:
            raise ValueError(
                f"Target timestamp {target} is not in the archive "
                f"({self._timestamps.iloc[0]} → {self._timestamps.iloc[-1]})."
            )
        idx = int(matches[0])
        return idx, self._scaled_features[idx]

    # ------------------------------------------------------------------
    def find_analogs(self, target_timestamp, K: int = 10) -> dict:
        """Find the K best analogs for ``target_timestamp``.

        Steps:
        1. Extract the forecast feature vector at the target timestamp.
        2. Exclude a 7-day window around the target (leakage control).
        3. Standardize features (done once at init via ``StandardScaler``).
        4. Compute normalized Euclidean distances to all remaining
           historical forecast vectors.
        5. Select the K smallest distances.
        6. Build the ensemble from the *observations* at those analog
           timestamps.

        Returns
        -------
        dict with keys:
            ``target_timestamp``   – the queried timestamp (ISO string)
            ``target_forecast``    – dict of fcst_* values at the target
            ``target_observation`` – dict of obs_* values at the target
            ``analogs``            – list of the K analog timestamps with
                                     their distances (closest first)
            ``ensemble_mean``      – mean of obs_* over the K analogs
            ``ensemble_median``    – median of obs_* over the K analogs
            ``p10``                – 10th percentile of obs_* over the K
            ``p90``                – 90th percentile of obs_* over the K
            ``ensemble_members``   – DataFrame of the K analog observations
        """
        target_idx, target_vector = self._target_row(target_timestamp)

        # 2. Exclusion window: drop target ± 7 days.
        delta_hours = (
            self._timestamps - self._timestamps.iloc[target_idx]
        ).dt.total_seconds() / 3600.0
        search_mask = delta_hours.abs() > EXCLUSION_WINDOW_HOURS
        search_idx = np.where(search_mask)[0]
        if len(search_idx) < K:
            raise ValueError(
                f"Only {len(search_idx)} candidates remain after the "
                f"7-day exclusion window; need at least K={K}."
            )

        # 4. Normalized Euclidean distance in standardized feature space.
        diffs = self._scaled_features[search_idx] - target_vector
        distances = np.linalg.norm(diffs, axis=1)

        # 5. K smallest distances.
        order = np.argsort(distances)[:K]
        analog_idx = search_idx[order]
        analog_distances = distances[order]

        analogs = [
            {
                "timestamp": str(self._timestamps.iloc[i]),
                "distance": float(d),
            }
            for i, d in zip(analog_idx, analog_distances)
        ]

        # 6. Ensemble from the observations at the analog timestamps.
        members = self.df.iloc[analog_idx][
            ["timestamp"] + OBSERVED_VARIABLES
        ].reset_index(drop=True)
        obs_values = members[OBSERVED_VARIABLES].to_numpy(dtype=float)

        def _summary(values: np.ndarray) -> dict:
            return {
                var: float(v)
                for var, v in zip(OBSERVED_VARIABLES, values)
            }

        target_row = self.df.iloc[target_idx]
        return {
            "target_timestamp": str(self._timestamps.iloc[target_idx]),
            "target_forecast": {
                var: float(target_row[var]) for var in FORECAST_FEATURES
            },
            "target_observation": {
                var: float(target_row[var]) for var in OBSERVED_VARIABLES
            },
            "analogs": analogs,
            "ensemble_members": members,
            "ensemble_mean": _summary(obs_values.mean(axis=0)),
            "ensemble_median": _summary(np.median(obs_values, axis=0)),
            "p10": _summary(np.percentile(obs_values, 10, axis=0)),
            "p90": _summary(np.percentile(obs_values, 90, axis=0)),
        }
