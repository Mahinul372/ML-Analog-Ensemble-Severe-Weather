"""Model Output Statistics (MOS) baseline — Phase B, reviewer revision 7.

Ordinary least-squares linear regression of observed gust (obs_gust)
on the four standardized forecast predictors (fcst_temp, fcst_wind,
fcst_gust, fcst_press). The scaler AND the regression are fitted on
each archive's TRAINING split only; the model is then applied to the
fixed test period. MOS is deterministic, so its CRPS equals its MAE
(single-member ensemble CRPS).

Protocol (identical to every other method): the shared nested splits
from data_prep.py, test high-gust events obs_gust >= 35 km/h.

Outputs (per nested regime <suffix>):
  metrics/mos_metrics_<suffix>.json
  metrics/nested_mos_errors_<suffix>.npz  (timestamps, raw_fcst, obs, mos)

Usage:
    python mos_baseline.py                 # all four nested archives
    python mos_baseline.py --regime nested_3yr
"""

import argparse
import json
import os

import numpy as np

from data_prep import (
    FEATURES,
    NESTED_REGIMES,
    STORM_THRESHOLD,
    TARGET,
    fit_feature_scaler,
    get_storm_events,
    load_and_split,
)

HERE = os.path.dirname(os.path.abspath(__file__))


def run_mos(regime):
    df, train_df, val_df, test_df, reg = load_and_split(regime=regime)
    assert reg == regime
    scaler = fit_feature_scaler(train_df)
    X_train = scaler.transform(train_df[FEATURES])
    # Design matrix with intercept.
    X_train_d = np.column_stack([np.ones(len(X_train)), X_train])
    y_train = train_df[TARGET].to_numpy(dtype=float)
    # OLS via least squares (deterministic, closed form).
    coef, *_ = np.linalg.lstsq(X_train_d, y_train, rcond=None)

    storms = get_storm_events(test_df, STORM_THRESHOLD)
    X_test = scaler.transform(storms[FEATURES])
    X_test_d = np.column_stack([np.ones(len(X_test)), X_test])
    mos = X_test_d @ coef
    raw = storms["fcst_gust"].to_numpy(dtype=float)
    obs = storms[TARGET].to_numpy(dtype=float)

    rmse = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))
    mae = lambda a, b: float(np.mean(np.abs(a - b)))
    metrics = {
        "method": "MOS (ordinary least-squares linear regression)",
        "regime": regime,
        "n_train_rows": int(len(train_df)),
        "n_test_high_gust_events": int(len(storms)),
        "predictors": FEATURES,
        "coefficients": {
            "intercept": float(coef[0]),
            **{f: float(c) for f, c in zip(FEATURES, coef[1:])},
        },
        "rmse_raw_nwp_kmh": round(rmse(raw, obs), 2),
        "mae_raw_nwp_kmh": round(mae(raw, obs), 2),
        "crps_raw_nwp_kmh": round(mae(raw, obs), 2),
        "rmse_mos_kmh": round(rmse(mos, obs), 2),
        "mae_mos_kmh": round(mae(mos, obs), 2),
        # Deterministic method: CRPS = MAE.
        "crps_mos_kmh": round(mae(mos, obs), 2),
        "rmse_skill_vs_raw_pct": round(100 * (rmse(raw, obs) - rmse(mos, obs)) / rmse(raw, obs), 1),
        "crps_skill_vs_raw_pct": round(100 * (mae(raw, obs) - mae(mos, obs)) / mae(raw, obs), 1),
    }
    os.makedirs(os.path.join(HERE, "metrics"), exist_ok=True)
    suffix = regime
    with open(os.path.join(HERE, "metrics", f"mos_metrics_{suffix}.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    np.savez(os.path.join(HERE, "metrics", f"nested_mos_errors_{suffix}.npz"),
             timestamps=np.array([str(t) for t in storms["timestamp"]]),
             raw_fcst=raw, obs=obs, mos=mos)
    print(f"{regime}: n_train={len(train_df)} events={len(storms)} | "
          f"Raw RMSE {metrics['rmse_raw_nwp_kmh']:.2f} CRPS {metrics['crps_raw_nwp_kmh']:.2f} | "
          f"MOS RMSE {metrics['rmse_mos_kmh']:.2f} MAE/CRPS {metrics['crps_mos_kmh']:.2f}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="MOS baseline (OLS)")
    parser.add_argument("--regime", type=str, default=None,
                        help="one nested regime (default: all four)")
    args = parser.parse_args()
    regimes = [args.regime] if args.regime else NESTED_REGIMES
    for r in regimes:
        if r not in NESTED_REGIMES:
            raise ValueError(f"unknown nested regime {r!r}")
        run_mos(r)


if __name__ == "__main__":
    main()
