"""Per-hour forecasts for the representative high-gust episode (Phase B, Fig 4).

Episode selection is by the PREDEFINED rule in data_prep.get_severe_episode
(contiguous run of test-window hours with obs_gust >= 35 km/h with the
highest observed peak; ties -> longest run), applied to the FIXED nested
test period 2025-11-01 -> 2026-11-01. The plotted window is the episode
padded by 24 h on each side.

Forecasts at every hour of the window, for the nested_8yr archive (the
primary configuration):
  * Observations + Raw NWP: straight from the archive.
  * MOS: OLS coefficients from metrics/mos_metrics_nested_8yr.json applied
    to the standardized forecast predictors (scaler fitted on nested_8yr
    training data only).
  * Traditional AnEn: 4-D analog search (K=15, +/-168 h, training only).
  * Deep-AnEn-Regression: saved nested_8yr weights, 16-D latent search.
  * Deep Analog (Hu et al., 2023): saved nested_8yr checkpoint, 20-D
    embedding search.

Output: metrics/nested_episode_forecasts.json — window bounds, episode
bounds, and per-hour arrays (timestamps, obs, raw, mos, and for each
ensemble method the mean + p5/p95 + members).

Usage: python nested_episode_forecasts.py
"""

import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from data_prep import (
    EXCLUSION_HOURS,
    FEATURES,
    K,
    STORM_THRESHOLD,
    TARGET,
    fit_feature_scaler,
    get_severe_episode,
    load_and_split,
)
from phase4_triplet import (
    SequenceStore,
    embed_positions,
    load_triplet_model,
    retrieve_analogs,
)
from traditional_anen import TraditionalAnalogEnsemble
from train_regression_nested import DeepAnalogRegressionEncoder

HERE = os.path.dirname(os.path.abspath(__file__))
REGIME = "nested_8yr"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df, train_df, val_df, test_df, regime = load_and_split(regime=REGIME)
    assert regime == REGIME
    ep = get_severe_episode(test_df, STORM_THRESHOLD, pad_hours=24)
    print(f"Episode: {ep['episode_start']} -> {ep['episode_end']} "
          f"({ep['n_runs']} runs in test window) | peak "
          f"{ep['peak_gust']:.1f} km/h at {ep['peak_timestamp']} | "
          f"window {ep['window_start']} -> {ep['window_end']}")
    window = df[(df["timestamp"] >= ep["window_start"])
                & (df["timestamp"] <= ep["window_end"])].reset_index(drop=True)
    print(f"Window hours: {len(window)}")
    feat_scaler = fit_feature_scaler(train_df)
    obs = window[TARGET].to_numpy(dtype=float)
    raw = window["fcst_gust"].to_numpy(dtype=float)
    ts = [str(t) for t in window["timestamp"]]

    # MOS from saved coefficients.
    with open(os.path.join(HERE, "metrics", f"mos_metrics_{REGIME}.json")) as f:
        mos_m = json.load(f)
    coef = np.array([mos_m["coefficients"]["intercept"]] +
                    [mos_m["coefficients"][f] for f in FEATURES])
    Xw = np.column_stack([np.ones(len(window)),
                          feat_scaler.transform(window[FEATURES])])
    mos = Xw @ coef

    # Traditional AnEn.
    trad = TraditionalAnalogEnsemble(df, regime=REGIME)
    trad_members = np.array([trad.find_analogs(t, K=K)["ensemble_members"]
                             for t in window["timestamp"]])

    # Deep-AnEn-Regression from saved nested weights.
    ckpt = torch.load(os.path.join(HERE, "models",
                                   f"deep_anen_regression_weights_{REGIME}.pth"),
                      map_location=device, weights_only=True)
    encoder = DeepAnalogRegressionEncoder().to(device)
    encoder.load_state_dict(ckpt["encoder_state_dict"])
    encoder.eval()
    X_train = torch.tensor(feat_scaler.transform(train_df[FEATURES]),
                           dtype=torch.float32).to(device)
    with torch.no_grad():
        train_z = encoder.encode(X_train).cpu().numpy()

    def regression_members(target_timestamp):
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
        return train_df.loc[cand_idx[order], TARGET].to_numpy(dtype=float)

    regr_members = np.array([regression_members(t) for t in window["timestamp"]])

    # Deep Analog (Hu et al., 2023) from saved nested checkpoint.
    model, _ = load_triplet_model(
        os.path.join(HERE, "models", f"deep_anen_triplet_weights_{REGIME}.pth"),
        device)
    store = SequenceStore(df, feat_scaler)
    train_pos = store.eligible_positions(train_df)
    train_emb = embed_positions(model, store, train_pos, device)
    trip_members, _, _ = retrieve_analogs(
        model, store, train_pos, list(window["timestamp"]), device, k=K,
        train_emb=train_emb)

    def pack(members):
        return {"mean": members.mean(axis=1).tolist(),
                "p5": np.percentile(members, 5, axis=1).tolist(),
                "p95": np.percentile(members, 95, axis=1).tolist(),
                "members": members.tolist()}

    out = {
        "regime": REGIME,
        "episode_start": str(ep["episode_start"]),
        "episode_end": str(ep["episode_end"]),
        "peak_timestamp": str(ep["peak_timestamp"]),
        "peak_gust_kmh": ep["peak_gust"],
        "window_start": str(ep["window_start"]),
        "window_end": str(ep["window_end"]),
        "n_runs_in_test_window": ep["n_runs"],
        "selection_rule": "contiguous run of test-window hours with obs_gust >= 35 km/h "
                          "with the highest observed peak gust; ties broken by longest run",
        "timestamps": ts,
        "obs_gust": obs.tolist(),
        "raw_nwp_gust": raw.tolist(),
        "mos_gust": mos.tolist(),
        "traditional_anen": pack(trad_members),
        "deep_anen_regression": pack(regr_members),
        "deep_analog": pack(trip_members),
    }
    with open(os.path.join(HERE, "metrics", "nested_episode_forecasts.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved metrics/nested_episode_forecasts.json ({len(window)} hours)")
    print(f"Window RMSE — Raw {np.sqrt(np.mean((raw-obs)**2)):.2f} | MOS "
          f"{np.sqrt(np.mean((mos-obs)**2)):.2f} | Trad "
          f"{np.sqrt(np.mean((trad_members.mean(axis=1)-obs)**2)):.2f} | Regr "
          f"{np.sqrt(np.mean((regr_members.mean(axis=1)-obs)**2)):.2f} | Trip "
          f"{np.sqrt(np.mean((trip_members.mean(axis=1)-obs)**2)):.2f}")


if __name__ == "__main__":
    main()
