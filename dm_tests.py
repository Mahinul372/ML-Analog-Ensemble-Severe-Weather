"""Diebold-Mariano (DM) tests for the analog-ensemble comparison.

The DM test (Diebold & Mariano, 1995) tests the null hypothesis of
equal predictive accuracy between two forecasts. For two forecasts with
per-event absolute errors e1, e2, the loss differential is
d_t = |e1_t| - |e2_t| and the test statistic is

    DM = dbar / sqrt(V_hat / T),

where V_hat is a Newey-West HAC estimator of the long-run variance of d
(Bartlett kernel, lag truncation h = floor(T^(1/3))). The two-sided
p-value comes from the standard normal. A negative mean differential
(dbar < 0) means the first forecast has the LOWER absolute error, i.e.
it is the better of the two; p < 0.05 means the gap is statistically
significant.

Variant documented here (this is the exact variant used for every
p-value in metrics/dm_tests.json and in the dual-regime table):
  * Loss: absolute error (the user's specified loss), per storm event.
  * Events are indexed by event order, not clock time (storm events are
    irregularly spaced); the HAC estimator allows for whatever
    autocorrelation the loss differential has across events.
  * Lag truncation h = floor(T^(1/3)) with the Bartlett kernel
    w_j = 1 - j/(h+1).
  * Two-sided p-value from N(0, 1). No small-sample (HLN) correction:
    with T = 130 (1-year) and T = 1,286 (long) the asymptotic
    approximation is the standard choice, and the variant is reported
    so the test is reproducible.

Pairs tested (both regimes), each vs Traditional AnEn:
  * Deep-AnEn-Regression vs Traditional AnEn
  * Deep Analog (Hu et al., 2023) vs Traditional AnEn
  * Raw NWP vs Traditional AnEn (reported as a computed p-value, never
    a placeholder)

1-year per-event errors are RECOMPUTED here from the saved 1-year
weights (never copied from a metric file). Long-regime errors are
loaded from the per-event .npz files saved by the three long-archive
model runs (same protocol, same evaluation code paths).

Usage:
    python dm_tests.py            # both regimes (long files must exist)
    python dm_tests.py --regime 1yr   # 1-year regime only
"""

import argparse
import json
import math
import os

import numpy as np
import pandas as pd
import properscoring
import torch
import torch.nn as nn
from scipy.stats import norm

from data_prep import (
    EXCLUSION_HOURS,
    FEATURES,
    K,
    TARGET,
    fit_feature_scaler,
    get_storm_events,
    load_and_split,
)
from traditional_anen import TraditionalAnalogEnsemble
from phase4_triplet import (
    SequenceStore,
    embed_positions,
    load_triplet_model,
    retrieve_analogs,
)

HERE = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# Diebold-Mariano test
# ---------------------------------------------------------------------------
def diebold_mariano(e1, e2):
    """DM test on absolute-error vectors. Returns a dict with the
    statistic, two-sided p-value, mean loss differential, n and lag h.

    d_t = |e1_t| - |e2_t|; negative dbar means forecast 1 is better.
    """
    e1 = np.asarray(e1, dtype=float)
    e2 = np.asarray(e2, dtype=float)
    assert e1.shape == e2.shape and e1.ndim == 1
    d = np.abs(e1) - np.abs(e2)
    T = len(d)
    dbar = float(d.mean())
    h = int(T ** (1.0 / 3.0))
    # Newey-West HAC long-run variance, 1/T normalisation, Bartlett kernel.
    dc = d - dbar
    gamma = []
    for j in range(h + 1):
        gamma.append(float(np.sum(dc[:T - j] * dc[j:]) / T))
    var_hat = gamma[0] + 2.0 * sum(
        (1.0 - j / (h + 1.0)) * gamma[j] for j in range(1, h + 1))
    if var_hat <= 0:
        # Degenerate case (identical forecasts): no evidence of a difference.
        return {"dm_statistic": 0.0, "p_value": 1.0, "mean_diff": dbar,
                "n": T, "lag_h": h, "var_hat": var_hat}
    dm = dbar / math.sqrt(var_hat / T)
    p = float(2.0 * (1.0 - norm.cdf(abs(dm))))
    return {"dm_statistic": float(dm), "p_value": p, "mean_diff": dbar,
            "n": T, "lag_h": h, "var_hat": float(var_hat)}


# ---------------------------------------------------------------------------
# 1-year per-event errors, recomputed from saved weights
# ---------------------------------------------------------------------------
class DeepAnalogRegressionEncoder(nn.Module):
    """Forecast (4) -> 32 -> ReLU -> Dropout -> 16-D latent embedding."""

    def __init__(self, input_dim=4, hidden_dim=32, latent_dim=16, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.head = nn.Linear(latent_dim, 1)

    def encode(self, x):
        return self.net(x)

    def forward(self, x):
        return self.head(self.encode(x))


def compute_1yr_errors():
    """Recompute per-event forecasts for all four methods on the 130
    1-year storm events, from the saved 1-year weights. Returns a dict
    of arrays: raw, obs, trad_mean, regression_mean, triplet_mean and
    the three member arrays (for CRPS cross-checks)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df, train_df, val_df, test_df, regime = load_and_split(
        os.path.join(HERE, "data", "atlanta_weather_archive_1yr.csv"))
    assert regime == "1yr"
    storms = get_storm_events(test_df)
    event_ts = list(storms["timestamp"])
    assert len(event_ts) == 130
    feat_scaler = fit_feature_scaler(train_df)

    # Traditional AnEn.
    trad = TraditionalAnalogEnsemble(df, regime="1yr")
    trad_members = np.array([trad.find_analogs(ts, K=K)["ensemble_members"]
                             for ts in event_ts])

    # Deep-AnEn-Regression from saved Phase 2 weights.
    ckpt = torch.load(os.path.join(HERE, "models",
                                   "deep_anen_regression_weights.pth"),
                      map_location=device, weights_only=True)
    encoder = DeepAnalogRegressionEncoder().to(device)
    encoder.load_state_dict(ckpt["encoder_state_dict"])
    encoder.eval()
    X_train = torch.tensor(feat_scaler.transform(train_df[FEATURES]),
                           dtype=torch.float32).to(device)
    with torch.no_grad():
        train_z = encoder.encode(X_train).cpu().numpy()

    def regression_find_analogs(target_timestamp):
        target = pd.to_datetime(target_timestamp, utc=True)
        row = df[df["timestamp"] == target]
        x = torch.tensor(feat_scaler.transform(row[FEATURES]),
                         dtype=torch.float32).to(device)
        with torch.no_grad():
            z = encoder.encode(x).cpu().numpy()[0]
        delta_h = (train_df["timestamp"] - target).dt.total_seconds() / 3600.0
        cand_idx = np.where((delta_h.abs() > EXCLUSION_HOURS).values)[0]
        dists = np.linalg.norm(train_z[cand_idx] - z, axis=1)
        order = np.argsort(dists)[:K]
        analog_idx = cand_idx[order]
        return train_df.loc[analog_idx, TARGET].to_numpy(dtype=float)

    regression_members = np.array([regression_find_analogs(ts) for ts in event_ts])

    # Deep Analog (Hu et al., 2023) from saved Phase 4 checkpoint.
    triplet_model, _ = load_triplet_model(
        os.path.join(HERE, "models", "deep_anen_triplet_weights.pth"), device)
    store = SequenceStore(df, feat_scaler)
    train_pos = store.eligible_positions(train_df)
    train_emb = embed_positions(triplet_model, store, train_pos, device)
    triplet_members, _, _ = retrieve_analogs(
        triplet_model, store, train_pos, event_ts, device, k=K, train_emb=train_emb)

    raw = storms["fcst_gust"].to_numpy(dtype=float)
    obs = storms[TARGET].to_numpy(dtype=float)
    return {
        "timestamps": [str(t) for t in event_ts],
        "raw": raw, "obs": obs,
        "trad_mean": trad_members.mean(axis=1),
        "regression_mean": regression_members.mean(axis=1),
        "triplet_mean": triplet_members.mean(axis=1),
        "trad_members": trad_members,
        "regression_members": regression_members,
        "triplet_members": triplet_members,
    }


def rmse_crps_from_members(raw, obs, members):
    rmse = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))
    return (rmse(raw, obs), float(np.mean(np.abs(raw - obs))),
            rmse(members.mean(axis=1), obs),
            float(np.mean(properscoring.crps_ensemble(obs, members))))


def run_dm_for_regime(err):
    """DM tests of each method vs Traditional AnEn, from per-event arrays."""
    obs = err["obs"]
    trad_err = err["trad_mean"] - obs
    out = {}
    for name, mean_fcst in [("Raw NWP", err["raw"]),
                            ("Deep-AnEn-Regression", err["regression_mean"]),
                            ("Deep Analog (Hu et al., 2023)", err["triplet_mean"])]:
        res = diebold_mariano(mean_fcst - obs, trad_err)
        out[name] = res
        print(f"  {name} vs Traditional AnEn: DM = {res['dm_statistic']:+.3f}, "
              f"p = {res['p_value']:.4g}, mean |e| diff = {res['mean_diff']:+.3f} km/h "
              f"(n={res['n']}, h={res['lag_h']})")
    return out


def main():
    parser = argparse.ArgumentParser(description="Diebold-Mariano tests")
    parser.add_argument("--regime", choices=["1yr", "long", "both"],
                        default="both")
    args = parser.parse_args()

    results = {"method": "Diebold-Mariano test on per-event absolute errors",
               "variant": "loss differential d = |e_method| - |e_traditional|; "
                          "Newey-West HAC long-run variance, Bartlett kernel, "
                          "lag h = floor(T^(1/3)); two-sided p-value from N(0,1); "
                          "negative mean diff means the method beats Traditional AnEn",
               "regimes": {}}

    if args.regime in ("1yr", "both"):
        print("1-year regime: recomputing per-event errors from saved weights ...")
        err = compute_1yr_errors()
        # Cross-check the recomputed scores against the saved metric files.
        rmse_raw, crps_raw, rmse_trad, crps_trad = rmse_crps_from_members(
            err["raw"], err["obs"], err["trad_members"])
        _, _, rmse_reg, crps_reg = rmse_crps_from_members(
            err["raw"], err["obs"], err["regression_members"])
        _, _, rmse_trip, crps_trip = rmse_crps_from_members(
            err["raw"], err["obs"], err["triplet_members"])
        print(f"  Recomputed 1-year: Raw {rmse_raw:.2f}/{crps_raw:.2f} | "
              f"Trad {rmse_trad:.2f}/{crps_trad:.2f} | "
              f"Reg {rmse_reg:.2f}/{crps_reg:.2f} | Trip {rmse_trip:.2f}/{crps_trip:.2f}")
        for got, want, name in [(rmse_raw, 13.23, "Raw RMSE"),
                                (crps_raw, 11.42, "Raw CRPS"),
                                (rmse_trad, 7.43, "Trad RMSE"),
                                (crps_trad, 4.15, "Trad CRPS"),
                                (rmse_reg, 7.86, "Reg RMSE"),
                                (crps_reg, 4.45, "Reg CRPS"),
                                (rmse_trip, 8.35, "Trip RMSE"),
                                (crps_trip, 5.15, "Trip CRPS")]:
            assert abs(got - want) < 0.05, f"{name}: recomputed {got} != saved {want}"
        print("  Cross-check passed: recomputed scores match the saved 1-year metrics.")
        np.savez(os.path.join(HERE, "metrics", "1yr_errors.npz"),
                 timestamps=np.array(err["timestamps"]), raw_fcst=err["raw"],
                 obs=err["obs"], trad_members=err["trad_members"],
                 regression_members=err["regression_members"],
                 triplet_members=err["triplet_members"])
        print("  Saved metrics/1yr_errors.npz")
        print("1-year DM tests (vs Traditional AnEn):")
        results["regimes"]["1yr"] = {
            "n_events": 130,
            "pairs_vs_traditional": run_dm_for_regime(err),
        }

    if args.regime in ("long", "both"):
        need = ["long_traditional_errors.npz", "long_regression_errors.npz",
                "long_triplet_errors.npz"]
        missing = [f for f in need
                   if not os.path.exists(os.path.join(HERE, "metrics", f))]
        if missing:
            print(f"Long regime skipped: missing {missing} "
                  "(run the long-archive model runs first).")
        else:
            print("Long regime: loading per-event arrays from the model runs ...")
            trad = np.load(os.path.join(HERE, "metrics",
                                        "long_traditional_errors.npz"))
            reg = np.load(os.path.join(HERE, "metrics",
                                       "long_regression_errors.npz"))
            trip = np.load(os.path.join(HERE, "metrics",
                                        "long_triplet_errors.npz"))
            assert (trad["timestamps"] == reg["timestamps"]).all()
            assert (trad["timestamps"] == trip["timestamps"]).all()
            assert (trad["obs"] == reg["obs"]).all()
            assert (trad["obs"] == trip["obs"]).all()
            err = {"raw": trad["raw_fcst"], "obs": trad["obs"],
                   "trad_mean": trad["members"].mean(axis=1),
                   "regression_mean": reg["members"].mean(axis=1),
                   "triplet_mean": trip["members"].mean(axis=1)}
            print(f"  Long events: {len(err['obs'])}")
            print("Long DM tests (vs Traditional AnEn):")
            results["regimes"]["long"] = {
                "n_events": int(len(err["obs"])),
                "pairs_vs_traditional": run_dm_for_regime(err),
            }

    out_path = os.path.join(HERE, "metrics", "dm_tests.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out_path}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
