"""Traditional Analog Ensemble (AnEn) benchmark — Phase 3.

Direct Euclidean-distance analog search in the 4-D standardized forecast
space (temperature, wind speed, wind gust, surface pressure), WITHOUT any
neural-network embedding. This is the classical AnEn of Delle Monache et
al. (2011, 2013) and the baseline against which the Phase 2 Deep-AnEn-Regression
(16-D PyTorch latent embedding) is compared.

The dataset, split, scaler, and evaluation subset are NOT defined here:
they are imported from data_prep.py, the single canonical protocol shared
by every experiment, so preprocessing is never recreated independently.

Scoring convention (consistent across all three methods):
  * RMSE on the ensemble mean (deterministic score).
  * CRPS via properscoring.crps_ensemble over the 15 members
    (probabilistic score).
  * For deterministic Raw NWP, CRPS is the mean absolute error — the
    exact equivalent of single-member ensemble CRPS.

Usage:
    python traditional_anen.py            # evaluate + save metrics JSON (1-year)
    python traditional_anen.py --demo     # also print one worked example
    python traditional_anen.py --archive data/atlanta_weather_archive_2018.csv \
        --metrics-out metrics/traditional_anen_metrics_8yr10mo.json \
        --errors-out metrics/long_traditional_errors.npz
        # long-archive run: any yearly file (or the data directory) loads
        # the nine yearly files concatenated; splits branch by regime (see data_prep)
"""

import argparse
import json
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import properscoring

from data_prep import (
    EXCLUSION_HOURS,
    FEATURES,
    K,
    STORM_THRESHOLD,
    TARGET,
    fit_feature_scaler,
    get_storm_events,
    load_archive,
    regime_for_path,
    split_archive,
)


@dataclass
class TraditionalAnalogEnsemble:
    """Classical AnEn: Euclidean search in standardized forecast space."""

    df: pd.DataFrame
    k: int = K
    exclusion_hours: int = EXCLUSION_HOURS
    regime: str = "1yr"
    train_df: pd.DataFrame = field(init=False)
    val_df: pd.DataFrame = field(init=False)
    test_df: pd.DataFrame = field(init=False)
    scaler: object = field(init=False)
    train_scaled: np.ndarray = field(init=False)

    def __post_init__(self):
        df = self.df.sort_values("timestamp").reset_index(drop=True)
        self.df = df
        self.train_df, self.val_df, self.test_df = split_archive(df, regime=self.regime)
        # Scaler fitted on the training split only (see data_prep).
        self.scaler = fit_feature_scaler(self.train_df)
        self.train_scaled = self.scaler.transform(self.train_df[FEATURES])

    def find_analogs(self, target_timestamp, K=None):
        """Top-k training-period analogs of `target_timestamp` in 4-D space.

        Returns a dict with the target forecast/observation, the k analog
        timestamps and distances, and the observed-gust ensemble
        (members, mean, median, p10, p90).
        """
        k = K or self.k
        target = pd.to_datetime(target_timestamp, utc=True)
        row = self.df[self.df["timestamp"] == target]
        if row.empty:
            raise ValueError(f"{target} is not in the archive.")
        x = self.scaler.transform(row[FEATURES])[0]

        # Exclude a +/-7-day window around the target (leakage control).
        delta_h = (self.train_df["timestamp"] - target).dt.total_seconds() / 3600.0
        mask = delta_h.abs() > self.exclusion_hours
        cand_idx = np.where(mask.values)[0]
        dists = np.linalg.norm(self.train_scaled[cand_idx] - x, axis=1)
        order = np.argsort(dists)[:k]
        analog_idx = cand_idx[order]

        members = self.train_df.loc[analog_idx, TARGET].to_numpy(dtype=float)
        return {
            "target_timestamp": str(target),
            "target_forecast_gust": float(row["fcst_gust"].iloc[0]),
            "target_observed_gust": float(row[TARGET].iloc[0]),
            "analogs": [
                {
                    "timestamp": str(self.train_df.loc[i, "timestamp"]),
                    "distance": float(d),
                }
                for i, d in zip(analog_idx, dists[order])
            ],
            "ensemble_members": members,
            "ensemble_mean": float(members.mean()),
            "ensemble_median": float(np.median(members)),
            "p10": float(np.percentile(members, 10)),
            "p90": float(np.percentile(members, 90)),
        }

    def evaluate(self, threshold=STORM_THRESHOLD, save_arrays=None):
        """Evaluate Raw NWP vs. Traditional AnEn on the shared storm events.

        The event list comes from data_prep.get_storm_events — the exact
        same timestamps used by the Deep-AnEn-Regression evaluation for
        the same regime. Returns (results_df, metrics_dict). When
        save_arrays is given, per-event raw forecasts, observations and
        ensemble members are saved as .npz (for DM tests / notebook use).
        """
        storms = get_storm_events(self.test_df, threshold)
        raw_fcst, obs, anen_mean, anen_members = [], [], [], []
        for _, event in storms.iterrows():
            res = self.find_analogs(event["timestamp"])
            raw_fcst.append(res["target_forecast_gust"])
            obs.append(res["target_observed_gust"])
            anen_mean.append(res["ensemble_mean"])
            anen_members.append(res["ensemble_members"])

        raw_fcst = np.array(raw_fcst)
        obs = np.array(obs)
        anen_mean = np.array(anen_mean)
        anen_members = np.array(anen_members)

        rmse = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))
        rmse_raw = rmse(raw_fcst, obs)
        rmse_anen = rmse(anen_mean, obs)
        # Raw NWP is deterministic: its CRPS is the mean absolute error,
        # the exact single-member equivalent of ensemble CRPS.
        crps_raw = float(np.mean(np.abs(raw_fcst - obs)))
        crps_anen = float(np.mean(properscoring.crps_ensemble(obs, anen_members)))

        results = pd.DataFrame(
            {
                "Method": ["Raw NWP", "Traditional AnEn"],
                "RMSE (km/h)": [rmse_raw, rmse_anen],
                "CRPS (km/h)": [crps_raw, crps_anen],
            }
        )
        metrics = {
            "regime": self.regime,
            "n_train_rows": int(len(self.train_df)),
            "n_test_storm_events": int(len(storms)),
            "rmse_raw_nwp_kmh": round(rmse_raw, 2),
            "rmse_traditional_anen_kmh": round(rmse_anen, 2),
            "rmse_skill_pct": round(100 * (rmse_raw - rmse_anen) / rmse_raw, 1),
            "crps_raw_nwp_kmh": round(crps_raw, 2),
            "crps_traditional_anen_kmh": round(crps_anen, 2),
            "crps_skill_pct": round(100 * (crps_raw - crps_anen) / crps_raw, 1),
        }
        if save_arrays is not None:
            np.savez(save_arrays,
                     timestamps=np.array([str(t) for t in storms["timestamp"]]),
                     raw_fcst=raw_fcst, obs=obs, members=anen_members)
            print(f"Saved per-event arrays to {save_arrays}")
        return results, metrics


def main():
    parser = argparse.ArgumentParser(description="Traditional AnEn benchmark")
    parser.add_argument("--demo", action="store_true",
                        help="print one worked analog example")
    parser.add_argument("--archive", type=str, default=None,
                        help="archive CSV (default: 1-year file; regime "
                             "branches by filename)")
    parser.add_argument("--metrics-out", type=str, default=None,
                        help="metrics JSON path (default: "
                             "metrics/traditional_anen_metrics.json)")
    parser.add_argument("--errors-out", type=str, default=None,
                        help="optional .npz path for per-event arrays")
    args = parser.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    archive_path = args.archive or os.path.join(here, "data",
                                                "atlanta_weather_archive_1yr.csv")
    regime = regime_for_path(archive_path)
    df = load_archive(archive_path)
    engine = TraditionalAnalogEnsemble(df, regime=regime)
    print(f"Archive: {archive_path} | regime: {regime}")
    print(f"Train: {len(engine.train_df)} | Val: {len(engine.val_df)} "
          f"| Test window: {len(engine.test_df)} rows")

    if args.demo:
        peak = engine.test_df.loc[engine.test_df[TARGET].idxmax(), "timestamp"]
        demo = engine.find_analogs(peak)
        print(f"\nDemo target {demo['target_timestamp']} | forecast "
              f"{demo['target_forecast_gust']:.1f} | observed "
              f"{demo['target_observed_gust']:.1f} km/h")
        print(f"Traditional AnEn mean {demo['ensemble_mean']:.1f} | "
              f"p10 {demo['p10']:.1f} | p90 {demo['p90']:.1f} km/h")
        print("Closest analogs:",
              [a["timestamp"][:16] for a in demo["analogs"][:5]])

    results, metrics = engine.evaluate(save_arrays=args.errors_out)
    print("\n" + results.to_string(index=False))
    print(f"\nRMSE skill: {metrics['rmse_skill_pct']:+.1f}% | "
          f"CRPS skill: {metrics['crps_skill_pct']:+.1f}%")

    out_path = args.metrics_out or os.path.join(here, "metrics",
                                                "traditional_anen_metrics.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
