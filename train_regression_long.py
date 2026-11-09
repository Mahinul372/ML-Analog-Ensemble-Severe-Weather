"""Retrain Deep-AnEn-Regression on the LONG archive (Option A: 2018-01-01 start).

Standalone equivalent of the Phase 2 notebook (build_notebook.py), but
for the long-archive regime from data_prep.py: train 2018-01-01 ->
2024-11-01 (59,928 rows), val 2024-11-02 -> 2025-11-01, test
2025-11-02 -> 2026-11-01 (1,286 storm events, obs_gust >= 35 km/h).

Architecture, optimizer and training are IDENTICAL to Phase 2:
DeepAnalogRegressionEncoder (Linear 4->32 -> ReLU -> Dropout(0.2) ->
Linear 32->16 latent + Linear 16->1 training head), Adam lr 1e-3, MSE
vs standardized obs_gust, 30 epochs, full-batch, seed 42. The only
difference is the training data (long archive instead of 1-year).

Analog search: K=15 nearest training-period embeddings, +/-168 h
exclusion, training pool only. Scores: RMSE on the ensemble mean,
CRPS via properscoring.crps_ensemble over the 15 members.

Outputs:
  models/deep_anen_regression_weights_10yr.pth  (binary checkpoint)
  models/deep_anen_regression_weights_10yr.json (text-safe state dict)
  metrics/deep_anen_regression_metrics_10yr.json
  metrics/long_regression_errors.npz  (per-event raw/obs/members, for DM tests)

Usage: python train_regression_long.py
"""

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


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"PyTorch {torch.__version__} | device: {device}")

    df, train_df, val_df, test_df, regime = load_and_split(regime="long")
    assert regime == "long"
    print(f"Archive: {len(df)} rows | Train: {len(train_df)} | "
          f"Val: {len(val_df)} | Test window: {len(test_df)} rows")

    feat_scaler = fit_feature_scaler(train_df)
    target_scaler = StandardScaler().fit(train_df[[TARGET]])

    def to_tensors(frame):
        X = torch.tensor(feat_scaler.transform(frame[FEATURES]), dtype=torch.float32)
        y = torch.tensor(target_scaler.transform(frame[[TARGET]]), dtype=torch.float32)
        return X, y

    X_train, y_train = to_tensors(train_df)
    X_val, y_val = to_tensors(val_df)
    print(f"Tensors: X_train {tuple(X_train.shape)} | X_val {tuple(X_val.shape)}")

    encoder = DeepAnalogRegressionEncoder().to(device)
    print(f"Trainable parameters: "
          f"{sum(p.numel() for p in encoder.parameters() if p.requires_grad):,}")
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
            print(f"Epoch {epoch:2d}/{EPOCHS} | train MSE {loss.item():.4f} | "
                  f"val MSE {val_loss:.4f}")
    print(f"\nFinal: train MSE {history['train'][-1]:.4f} | "
          f"val MSE {history['val'][-1]:.4f} (standardized gust units)")

    # --- Latent embeddings for the training archive -------------------------
    encoder.eval()
    with torch.no_grad():
        train_z = encoder.encode(X_train_d).cpu().numpy()
    print(f"Latent training archive: {train_z.shape}")

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

    # --- Evaluate on the long test storm events ------------------------------
    storms = get_storm_events(test_df, STORM_THRESHOLD)
    print(f"Test storm events (obs_gust >= 35 km/h): {len(storms)}")
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
    rmse_raw = rmse(raw_fcst, obs)
    rmse_anen = rmse(anen_mean, obs)
    crps_raw = float(np.mean(properscoring.crps_ensemble(obs, raw_fcst[:, None])))
    crps_anen = float(np.mean(properscoring.crps_ensemble(obs, anen_members)))
    skill_rmse = 100 * (rmse_raw - rmse_anen) / rmse_raw
    skill_crps = 100 * (crps_raw - crps_anen) / crps_raw
    print(f"\nRaw NWP: RMSE {rmse_raw:.2f} | CRPS {crps_raw:.2f}")
    print(f"Deep-AnEn-Regression: RMSE {rmse_anen:.2f} | CRPS {crps_anen:.2f}")
    print(f"RMSE skill: {skill_rmse:+.1f}% | CRPS skill: {skill_crps:+.1f}%")

    # --- Save outputs ---------------------------------------------------------
    os.makedirs(os.path.join(HERE, "models"), exist_ok=True)
    os.makedirs(os.path.join(HERE, "metrics"), exist_ok=True)
    torch.save({
        "encoder_state_dict": encoder.state_dict(),
        "config": {"input_dim": 4, "hidden_dim": 32, "latent_dim": 16,
                   "dropout": 0.2, "epochs": EPOCHS, "lr": LR, "K": K,
                   "regime": "long",
                   "train_start": "2018-01-01", "train_end": "2024-11-01",
                   "test_start": "2025-11-02"},
        "feat_scaler_mean": feat_scaler.mean_.tolist(),
        "feat_scaler_scale": feat_scaler.scale_.tolist(),
        "target_scaler_mean": target_scaler.mean_.tolist(),
        "target_scaler_scale": target_scaler.scale_.tolist(),
        "history": history,
    }, os.path.join(HERE, "models", "deep_anen_regression_weights_10yr.pth"))

    weights_json = {k: v.cpu().tolist() for k, v in encoder.state_dict().items()}
    with open(os.path.join(HERE, "models",
                           "deep_anen_regression_weights_10yr.json"), "w") as f:
        json.dump({"config": {"input_dim": 4, "hidden_dim": 32, "latent_dim": 16,
                              "dropout": 0.2, "epochs": EPOCHS, "lr": LR, "K": K,
                              "regime": "long"},
                   "state_dict": weights_json}, f)

    metrics = {
        "regime": "long",
        "n_train_rows": int(len(train_df)),
        "n_test_storm_events": int(len(storms)),
        "rmse_raw_nwp_kmh": round(rmse_raw, 2),
        "rmse_deep_anen_regression_kmh": round(rmse_anen, 2),
        "rmse_skill_pct": round(skill_rmse, 1),
        "crps_raw_nwp_kmh": round(crps_raw, 2),
        "crps_deep_anen_regression_kmh": round(crps_anen, 2),
        "crps_skill_pct": round(skill_crps, 1),
    }
    with open(os.path.join(HERE, "metrics",
                           "deep_anen_regression_metrics_10yr.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    np.savez(os.path.join(HERE, "metrics", "long_regression_errors.npz"),
             timestamps=np.array([str(t) for t in storms["timestamp"]]),
             raw_fcst=raw_fcst, obs=obs, members=anen_members)
    print("\nSaved: models/deep_anen_regression_weights_10yr.pth/.json, "
          "metrics/deep_anen_regression_metrics_10yr.json, "
          "metrics/long_regression_errors.npz")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
