"""Retrain Deep-AnEn-Regression on a NESTED archive (Phase B).

Standalone equivalent of the Phase 2 notebook (build_notebook.py) and
of train_regression_long.py, but for one nested regime from
data_prep.py: fixed test 2025-11-01 -> 2026-11-01 (1,286 high-gust
events, obs_gust >= 35 km/h), fixed validation 2025-10-01 ->
2025-10-31, training ends 2025-09-30; only the training start varies
(nested_1yr 2024-11-01, nested_3yr 2022-11-01, nested_5yr 2020-11-01,
nested_8yr effective 2018-01-01).

Architecture, optimizer and training are IDENTICAL to Phase 2:
DeepAnalogRegressionEncoder (Linear 4->32 -> ReLU -> Dropout(0.2) ->
Linear 32->16 latent + Linear 16->1 training head), Adam lr 1e-3, MSE
vs standardized obs_gust, 30 epochs, full-batch, seed 42. The only
difference is the training data.

Analog search: K=15 nearest training-period embeddings, +/-168 h
exclusion, training pool only. Scores: RMSE on the ensemble mean,
CRPS via properscoring.crps_ensemble over the 15 members.

Outputs (per nested regime <suffix>):
  models/deep_anen_regression_weights_<suffix>.pth  (binary checkpoint)
  models/deep_anen_regression_weights_<suffix>.json (text-safe state dict)
  metrics/deep_anen_regression_metrics_<suffix>.json
  metrics/nested_regression_errors_<suffix>.npz  (timestamps, raw, obs, members)
  metrics/deep_anen_regression_history_<suffix>.json  (train/val loss)

Usage:
    python train_regression_nested.py --regime nested_1yr
    python train_regression_nested.py                 # all four nested archives
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import properscoring
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from data_prep import (
    EXCLUSION_HOURS,
    FEATURES,
    K,
    NESTED_REGIMES,
    STORM_THRESHOLD,
    TARGET,
    fit_feature_scaler,
    get_storm_events,
    load_and_split,
)

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 42
EPOCHS = 30
LR = 1e-3


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


def run_regression(regime):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{regime}] PyTorch {torch.__version__} | device: {device}")

    df, train_df, val_df, test_df, reg = load_and_split(regime=regime)
    assert reg == regime
    print(f"[{regime}] Archive: {len(df)} rows | Train: {len(train_df)} | "
          f"Val: {len(val_df)} | Test window: {len(test_df)} rows")

    feat_scaler = fit_feature_scaler(train_df)
    target_scaler = StandardScaler().fit(train_df[[TARGET]])

    def to_tensors(frame):
        X = torch.tensor(feat_scaler.transform(frame[FEATURES]), dtype=torch.float32)
        y = torch.tensor(target_scaler.transform(frame[[TARGET]]), dtype=torch.float32)
        return X, y

    X_train, y_train = to_tensors(train_df)
    X_val, y_val = to_tensors(val_df)

    encoder = DeepAnalogRegressionEncoder().to(device)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=LR)
    criterion = nn.MSELoss()
    X_train_d, y_train_d = X_train.to(device), y_train.to(device)
    X_val_d, y_val_d = X_val.to(device), y_val.to(device)

    history = {"train": [], "val": []}
    for epoch in range(1, EPOCHS + 1):
        encoder.train()
        optimizer.zero_grad()
        loss = criterion(encoder(X_train_d), y_train_d)
        loss.backward()
        optimizer.step()
        encoder.eval()
        with torch.no_grad():
            val_loss = criterion(encoder(X_val_d), y_val_d).item()
        history["train"].append(loss.item())
        history["val"].append(val_loss)
        if epoch == 1 or epoch % 5 == 0:
            print(f"[{regime}] Epoch {epoch:2d}/{EPOCHS} | train MSE {loss.item():.4f} | "
                  f"val MSE {val_loss:.4f}")
    print(f"[{regime}] Final: train MSE {history['train'][-1]:.4f} | "
          f"val MSE {history['val'][-1]:.4f} (standardized gust units)")

    encoder.eval()
    with torch.no_grad():
        train_z = encoder.encode(X_train_d).cpu().numpy()
    print(f"[{regime}] Latent training archive: {train_z.shape}")

    def find_analogs(target_timestamp, K=K):
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
        members = train_df.loc[analog_idx, TARGET].to_numpy(dtype=float)
        return {
            "target_timestamp": str(target),
            "target_forecast_gust": float(row["fcst_gust"].iloc[0]),
            "target_observed_gust": float(row[TARGET].iloc[0]),
            "ensemble_members": members,
            "ensemble_mean": float(members.mean()),
        }

    storms = get_storm_events(test_df, STORM_THRESHOLD)
    print(f"[{regime}] Test high-gust events (obs_gust >= 35 km/h): {len(storms)}")
    raw_fcst, obs, anen_mean, anen_members = [], [], [], []
    for _, event in storms.iterrows():
        res = find_analogs(event["timestamp"], K=K)
        raw_fcst.append(res["target_forecast_gust"])
        obs.append(res["target_observed_gust"])
        anen_mean.append(res["ensemble_mean"])
        anen_members.append(res["ensemble_members"])
    raw_fcst = np.array(raw_fcst)
    obs = np.array(obs)
    anen_mean = np.array(anen_mean)
    anen_members = np.array(anen_members)

    rmse = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))
    mae = lambda a, b: float(np.mean(np.abs(a - b)))
    rmse_raw = rmse(raw_fcst, obs)
    rmse_anen = rmse(anen_mean, obs)
    crps_raw = float(np.mean(properscoring.crps_ensemble(obs, raw_fcst[:, None])))
    crps_anen = float(np.mean(properscoring.crps_ensemble(obs, anen_members)))
    print(f"[{regime}] Raw NWP: RMSE {rmse_raw:.2f} | CRPS {crps_raw:.2f}")
    print(f"[{regime}] Deep-AnEn-Regression: RMSE {rmse_anen:.2f} | CRPS {crps_anen:.2f} | "
          f"MAE {mae(anen_mean, obs):.2f}")

    suffix = regime
    os.makedirs(os.path.join(HERE, "models"), exist_ok=True)
    os.makedirs(os.path.join(HERE, "metrics"), exist_ok=True)
    torch.save({
        "encoder_state_dict": encoder.state_dict(),
        "config": {"input_dim": 4, "hidden_dim": 32, "latent_dim": 16,
                   "dropout": 0.2, "epochs": EPOCHS, "lr": LR, "K": K,
                   "regime": regime,
                   "train_start": str(train_df["timestamp"].min()),
                   "train_end": str(train_df["timestamp"].max()),
                   "test_start": str(test_df["timestamp"].min())},
        "feat_scaler_mean": feat_scaler.mean_.tolist(),
        "feat_scaler_scale": feat_scaler.scale_.tolist(),
        "target_scaler_mean": target_scaler.mean_.tolist(),
        "target_scaler_scale": target_scaler.scale_.tolist(),
        "history": history,
    }, os.path.join(HERE, "models", f"deep_anen_regression_weights_{suffix}.pth"))

    weights_json = {k: v.cpu().tolist() for k, v in encoder.state_dict().items()}
    with open(os.path.join(HERE, "models",
                           f"deep_anen_regression_weights_{suffix}.json"), "w") as f:
        json.dump({"config": {"input_dim": 4, "hidden_dim": 32, "latent_dim": 16,
                              "dropout": 0.2, "epochs": EPOCHS, "lr": LR, "K": K,
                              "regime": regime},
                   "state_dict": weights_json}, f)

    metrics = {
        "regime": regime,
        "n_train_rows": int(len(train_df)),
        "n_test_high_gust_events": int(len(storms)),
        "rmse_raw_nwp_kmh": round(rmse_raw, 2),
        "rmse_deep_anen_regression_kmh": round(rmse_anen, 2),
        "mae_deep_anen_regression_kmh": round(mae(anen_mean, obs), 2),
        "rmse_skill_pct": round(100 * (rmse_raw - rmse_anen) / rmse_raw, 1),
        "crps_raw_nwp_kmh": round(crps_raw, 2),
        "crps_deep_anen_regression_kmh": round(crps_anen, 2),
        "crps_skill_pct": round(100 * (crps_raw - crps_anen) / crps_raw, 1),
        "train_loss_final": round(history["train"][-1], 4),
        "val_loss_final": round(history["val"][-1], 4),
    }
    with open(os.path.join(HERE, "metrics",
                           f"deep_anen_regression_metrics_{suffix}.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(HERE, "metrics",
                           f"deep_anen_regression_history_{suffix}.json"), "w") as f:
        json.dump({"regime": regime, "history": history}, f)

    np.savez(os.path.join(HERE, "metrics", f"nested_regression_errors_{suffix}.npz"),
             timestamps=np.array([str(t) for t in storms["timestamp"]]),
             raw_fcst=raw_fcst, obs=obs, members=anen_members)
    print(f"[{regime}] Saved weights/metrics/history/errors with suffix {suffix}")
    print(json.dumps(metrics, indent=2))
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Deep-AnEn-Regression nested retrain")
    parser.add_argument("--regime", type=str, default=None,
                        help="one nested regime (default: all four)")
    args = parser.parse_args()
    regimes = [args.regime] if args.regime else NESTED_REGIMES
    for r in regimes:
        if r not in NESTED_REGIMES:
            raise ValueError(f"unknown nested regime {r!r}")
        run_regression(r)


if __name__ == "__main__":
    main()
