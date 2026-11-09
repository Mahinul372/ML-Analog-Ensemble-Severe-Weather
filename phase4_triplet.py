"""Phase 4: Hu et al. (2023) Deep Analog — LSTM + Reverse Analog triplets.

Paper: Hu, W., Cervone, G., Young, G., et al. (2023). "Machine Learning
Weather Analogs for Near-Surface Variables." Boundary-Layer Meteorology,
186, 711-735. DOI: https://doi.org/10.1007/s10546-022-00779-6
Official implementation: https://github.com/uga-gaim/DeepAnalog

Methodology verified against the official implementation (files
DeepAnalogs/Embeddings.py, train.py, AnEnDataset.py, utils.py, and
Examples/example.yaml) — NOT assumed. Verified facts:
  * Architecture (Embeddings.py, EmbeddingLSTM): nn.LSTM over the
    sequence dimension, last-timestep output, then Linear(hidden, output)
    when fc_last=True. NO L2 normalization anywhere.
  * Loss (train.py): torch.nn.TripletMarginLoss(margin=train_margin),
    official default train_margin = 0.9 (example.yaml and utils.py
    add_default_values). PyTorch default p=2 (Euclidean). Margin 1.0 is
    NOT the official value.
  * Optimizer: Adam. LR scheduler: ReduceLROnPlateau(factor=0.2,
    patience=5, min_lr=1e-5). Early stopping monitors validation loss
    (EarlyStopping.py: improvement = loss < best + delta, delta=0).
  * Reverse Analog / fitness sampling (AnEnDataset.py _select_fitness):
    positives = the num_analogs entries with the smallest observation
    distance to the anchor; negatives are sampled WITHOUT replacement
    from the remaining entries with probability proportional to fitness,
    where fitness is the normalized inverse observation distance — i.e.
    observationally CLOSE non-positives (hard negatives) are sampled
    most often. _check_and_add keeps a triplet only if
    d_n - d_p <= dataset_margin; the official default dataset_margin is
    NaN, which disables that filter.
  * Observation distance (utils.py sort_distance): mean absolute
    difference of the (optionally scaled) observations; default
    y_scaler_type=None and julian_weight=0.0, i.e. raw |d_obs|.
  * Time window (AnEnDatasetWithTimeWindow): the input sequence is a
    window of forecast steps centred on a lead time; the official
    example uses lstm_radius=1 (3 steps).

Adaptations required by our single-location hourly dataset (documented,
not hidden):
  * The paper's sequence dimension is FORECAST LEAD TIMES (one forecast
    issued at many lead times). Our archive has one forecast per hour,
    so the sequence is a LOOKBACK window of consecutive hourly forecasts
    ending at the target hour. Sequence length SEQ_LEN=24: 24 hourly
    steps span the full diurnal cycle — the dominant timescale for
    near-surface temperature, wind, gust and pressure — and is the
    standard lookback for hourly meteorological sequence models. The
    official example's 3-step window would carry almost no temporal
    information at hourly resolution.
  * Every sequence lies entirely inside the split of its timestamp
    (train/val/test) and uses only records at or before the target hour:
    no future information, no split-boundary crossing.
  * The paper's default forecast scaler is MinMax; our shared protocol
    (data_prep.py) mandates the existing StandardScaler fitted on the
    training split only. The observation distance needs no scaler: with
    a single target variable it is the raw |d_obs_gust| in km/h, exactly
    as in the official code with y_scaler_type=None.
  * Positive group size = 15 (= retrieval K) instead of the paper's
    example of 21, keeping triplet positives consistent with the
    K=15 retrieval used in every phase. fitness_num_negative = 1
    (one negative per positive; the official standard triplet).
  * Anchors per epoch: a random subset of 2,000 training anchors
    (resampled every epoch) instead of all 6,864, for CPU training time;
    each anchor contributes all 15 positives. This is standard SGD
    subsampling; coverage accumulates across epochs.
  * Max epochs = 100 (official example: 20) so early stopping with
    patience 15 can actually engage.
  * The paper searches analogs over lead-time windows; we retrieve from
    the training archive only, with the same +/-168 h exclusion as
    Phases 2-3.

Shared protocol (mandatory, from data_prep.py — never recreated here):
dataset, chronological splits (train 6,888 / val 384 / test 1,488),
train-only StandardScaler, the 130 test storm events, K=15, +/-168 h
exclusion. Phase 2/3 experiments and their saved metrics are untouched.

Usage:
    python phase4_triplet.py            # train + evaluate + save metrics
    python phase4_triplet.py --epochs 60 --anchors-per-epoch 1500
"""

import argparse
import json
import os
import random

import numpy as np
import pandas as pd
import properscoring
import torch
import torch.nn as nn

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

# ---------------------------------------------------------------------------
# Configuration — paper values first, adaptations documented above.
# ---------------------------------------------------------------------------
PAPER = "Hu et al. (2023), Machine Learning Weather Analogs for Near-Surface Variables, Boundary-Layer Meteorology 186, 711-735, DOI: 10.1007/s10546-022-00779-6"
PAPER_REPO = "https://github.com/uga-gaim/DeepAnalog"

SEQ_LEN = 24            # lookback window (hours); see adaptation note above
HIDDEN = 20             # paper Sec. 2.2.1
LAYERS = 3              # paper Sec. 2.2.1 (three stacked LSTM layers)
DROPOUT = 0.01          # paper Sec. 2.2.1
EMBED_DIM = 20          # paper Sec. 2.2.1 (linear output layer)
LR = 0.005              # paper training configuration
MARGIN = 0.9            # official default train_margin (train.py/utils.py)
PATIENCE = 15           # early stopping patience (epochs)
MAX_EPOCHS = 100
BATCH = 256
NUM_ANALOGS = 15        # positive group size (= retrieval K)
FITNESS_NUM_NEGATIVE = 1
ANCHORS_PER_EPOCH = 2000
SEED = 42


# ---------------------------------------------------------------------------
# Small helpers.
# ---------------------------------------------------------------------------
def _to_naive_np_datetime64(ts):
    """Timestamps as naive datetime64 (all archive times are UTC).

    Avoids the numpy "no explicit representation of timezones" warning
    when subtracting a tz-aware Timestamp from a datetime64 array; the
    instants (and hence all hour differences) are unchanged.
    """
    return np.datetime64(pd.Timestamp(ts).tz_localize(None))


# ---------------------------------------------------------------------------
# Model — faithful to official EmbeddingLSTM (fc_last=True).
# ---------------------------------------------------------------------------
class DeepAnalogLSTM(nn.Module):
    """LSTM (3 layers, 20 hidden) + linear layer -> 20-D embedding.

    Mirrors DeepAnalog/DeepAnalogs/Embeddings.py::EmbeddingLSTM with
    fc_last=True: LSTM over the sequence, last-timestep hidden state,
    Linear(hidden, output). No regression head, no L2 normalization.
    """

    def __init__(self, input_features=4, hidden_features=HIDDEN,
                 hidden_layers=LAYERS, output_features=EMBED_DIM,
                 dropout=DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_features,
            hidden_size=hidden_features,
            num_layers=hidden_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.fc = nn.Linear(in_features=hidden_features,
                            out_features=output_features)

    def forward(self, x):
        # x: (batch, SEQ_LEN, 4) — already transposed to [batch, time, features]
        output, _ = self.lstm(x)
        output = output.select(1, -1)  # last timestep, as in the official code
        return self.fc(output)


# ---------------------------------------------------------------------------
# Sequences — lookback windows, strictly inside their own split.
# ---------------------------------------------------------------------------
class SequenceStore:
    """Precomputed (n, SEQ_LEN, 4) standardized sequences for the archive.

    Position i holds the window ending at archive row i (rows i-23..i).
    A position is eligible only if its whole window lies inside the same
    split as row i (no future records, no split-boundary crossing).
    """

    def __init__(self, df, scaler):
        self.df = df
        scaled = scaler.transform(df[FEATURES]).astype(np.float32)
        n = len(df)
        seqs = np.zeros((n, SEQ_LEN, len(FEATURES)), dtype=np.float32)
        for i in range(n):
            if i >= SEQ_LEN - 1:
                seqs[i] = scaled[i - SEQ_LEN + 1: i + 1]
        self.seqs = seqs
        # Hourly contiguity of the archive (asserted, not assumed).
        diffs = df["timestamp"].diff().dropna()
        assert (diffs == pd.Timedelta(hours=1)).all(), "archive is not hourly-complete"

    def eligible_positions(self, split_df):
        """Archive positions of split_df rows whose window is inside the split."""
        pos = np.where(self.df["timestamp"].isin(split_df["timestamp"]).to_numpy())[0]
        start_pos = pos.min()
        return pos[pos - start_pos >= SEQ_LEN - 1]


# ---------------------------------------------------------------------------
# Reverse Analog triplet sampling — official `_select_fitness` logic.
# ---------------------------------------------------------------------------
def sample_triplets(anchor_pos, candidate_pos, obs, timestamps, rng,
                    n_positives=NUM_ANALOGS,
                    n_negatives=FITNESS_NUM_NEGATIVE):
    """Build (anchor, positive, negative) position triplets.

    For each anchor: eligible candidates exclude the +/-168 h window
    around the anchor (and the anchor itself). Candidates are ranked by
    |obs[cand] - obs[anchor]| ascending. Positives = the n_positives
    closest. Negatives are sampled WITHOUT replacement from the rest with
    probability proportional to fitness = normalized inverse observation
    distance (official _select_fitness: fitness_i = (sum(d) - d_i) /
    sum(sum(d) - d_j), so observationally close non-positives — hard
    negatives — are sampled most often). dataset_margin is NaN in the
    official default, i.e. no triplet is discarded by the margin filter.
    Returns an (n_triplets, 3) int array of archive positions.
    """
    timestamps = np.asarray(timestamps, dtype="datetime64[ns]")
    cand_obs = obs[candidate_pos]
    cand_time = timestamps[candidate_pos]
    triplets = []
    for a in anchor_pos:
        dt_h = np.abs((cand_time - timestamps[a]).astype("timedelta64[h]").astype(float))
        eligible = np.where(dt_h > EXCLUSION_HOURS)[0]
        if len(eligible) <= n_positives:
            continue
        d = np.abs(cand_obs[eligible] - obs[a])
        order = np.argsort(d)
        pos_local = order[:n_positives]
        neg_pool_local = order[n_positives:]
        neg_d = d[neg_pool_local]
        # Fitness-proportionate sampling without replacement (official logic).
        inv = neg_d.sum() - neg_d
        fitness = inv / inv.sum()
        chosen = []
        for _ in range(n_positives * n_negatives):
            csum = np.cumsum(fitness)
            idx = int(np.digitize(rng.random(), csum))
            idx = min(idx, len(fitness) - 1)
            chosen.append(neg_pool_local[idx])
            fitness[idx] = 0.0
            s = fitness.sum()
            if s <= 0:
                break
            fitness = fitness / s
        for p_local, n_local in zip(
                [pos_local[i] for i in range(n_positives)],
                chosen[:n_positives * n_negatives]):
            triplets.append((a, eligible[p_local], eligible[n_local]))
    return np.array(triplets, dtype=np.int64)


# ---------------------------------------------------------------------------
# Training.
# ---------------------------------------------------------------------------
def train_triplet_model(store, train_pos, val_pos, obs, timestamps,
                        device, max_epochs=MAX_EPOCHS,
                        anchors_per_epoch=ANCHORS_PER_EPOCH, seed=SEED,
                        verbose=True, ckpt_path=None, scaler=None,
                        val_anchors_max=500, regime=None, resume_from=None):
    """Train DeepAnalogLSTM with RA triplets; early stopping on val loss.

    Validation triplets are built ONCE (fixed seed) from validation
    anchors with candidates from the TRAINING pool — mirroring retrieval,
    where test targets also search the training archive. At most
    val_anchors_max validation anchors are used (the 1-year regime has
    361, so this changes nothing there; the long regime has ~8,737, and
    capping at 500 anchors / 7,500 triplets keeps per-epoch validation
    time comparable to Phase 4's 5,415-triplet set while the mean loss
    stays stable). Returns (best_model, history dict). When ckpt_path
    is given, the best model so far is saved to disk on EVERY
    improvement, so a killed run can still be evaluated from its best
    checkpoint.

    resume_from: path of a checkpoint saved on a previous (killed) run
    of the SAME regime/config. The model weights, loss history, best
    epoch and best validation loss are restored and training continues
    from the next epoch. This is near-exact: per-epoch triplet sampling
    depends only on (seed + epoch), and the fixed validation set is
    rebuilt identically (seed 123); only the Adam/scheduler state is
    fresh (documented, not hidden). The early-stopping counter is
    restored as epochs_trained - best_epoch.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model = DeepAnalogLSTM().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.TripletMarginLoss(margin=MARGIN)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.2, patience=5, min_lr=1e-5)

    # Resume: restore weights + history + early-stopping state.
    history = {"train_loss": [], "val_loss": []}
    best_val = float("inf")
    best_state = None
    best_epoch = 0
    counter = 0
    start_epoch = 1
    if resume_from is not None and os.path.exists(resume_from):
        ckpt = torch.load(resume_from, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        history = ckpt["history"]
        best_val = history["best_val_loss"]
        best_epoch = history["best_epoch"]
        start_epoch = history["epochs_trained"] + 1
        counter = history["epochs_trained"] - best_epoch
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if verbose:
            print(f"Resumed from {resume_from}: best val {best_val:.4f} "
                  f"(epoch {best_epoch}), continuing from epoch {start_epoch}, "
                  f"no-improvement counter {counter} (optimizer/scheduler state fresh).")
        if start_epoch > max_epochs:
            print("Resume point is already at/past max epochs; evaluating the checkpoint.")
            model.eval()
            history["early_stopped"] = counter >= PATIENCE
            return model, history

    # Fixed validation triplet set (stable early-stopping signal).
    val_rng = np.random.default_rng(123)
    if len(val_pos) > val_anchors_max:
        val_anchors = val_rng.choice(val_pos, size=val_anchors_max, replace=False)
    else:
        val_anchors = val_pos
    val_triplets = sample_triplets(val_anchors, train_pos, obs, timestamps, val_rng)
    if verbose:
        print(f"Validation triplets (fixed): {len(val_triplets)}")

    def run_triplets(triplets, train_mode):
        model.train() if train_mode else model.eval()
        losses = []
        idx = np.arange(len(triplets))
        if train_mode:
            np.random.shuffle(idx)
        with torch.set_grad_enabled(train_mode):
            for s in range(0, len(idx), BATCH):
                b = triplets[idx[s:s + BATCH]]
                a = torch.tensor(store.seqs[b[:, 0]], device=device)
                p = torch.tensor(store.seqs[b[:, 1]], device=device)
                n = torch.tensor(store.seqs[b[:, 2]], device=device)
                loss = loss_fn(model(a), model(p), model(n))
                if train_mode:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                losses.append(loss.item())
        return float(np.mean(losses))

    epochs_trained = start_epoch - 1
    for epoch in range(start_epoch, max_epochs + 1):
        rng = np.random.default_rng(seed + epoch)
        anchors = rng.choice(train_pos, size=min(anchors_per_epoch, len(train_pos)),
                             replace=False)
        triplets = sample_triplets(anchors, train_pos, obs, timestamps, rng)
        train_loss = run_triplets(triplets, train_mode=True)
        val_loss = run_triplets(val_triplets, train_mode=False)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        epochs_trained = epoch
        if val_loss < best_val:  # official EarlyStopping: loss < best + delta(0)
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            counter = 0
            history["best_epoch"] = epoch
            history["best_val_loss"] = best_val
            history["epochs_trained"] = epoch
            if ckpt_path is not None:
                save_checkpoint(model, history, ckpt_path, scaler, regime=regime)
        else:
            counter += 1
        scheduler.step(val_loss)
        if verbose:
            print(f"Epoch {epoch:3d}/{max_epochs} | train {train_loss:.4f} | "
                  f"val {val_loss:.4f} | best val {best_val:.4f} (ep {best_epoch})")
        if counter >= PATIENCE:
            if verbose:
                print(f"Early stopping after {PATIENCE} epochs without improvement.")
            break

    model.load_state_dict(best_state)
    model.eval()
    history["best_epoch"] = best_epoch
    history["best_val_loss"] = best_val
    history["epochs_trained"] = epochs_trained
    history["early_stopped"] = counter >= PATIENCE
    return model, history


# ---------------------------------------------------------------------------
# Retrieval + evaluation on the shared 130 storm events.
# ---------------------------------------------------------------------------
def embed_positions(model, store, positions, device, batch=512):
    """Embeddings for archive positions, in order. Returns (n, EMBED_DIM)."""
    embs = []
    with torch.no_grad():
        for s in range(0, len(positions), batch):
            x = torch.tensor(store.seqs[positions[s:s + batch]], device=device)
            embs.append(model(x).cpu().numpy())
    return np.concatenate(embs, axis=0)


def retrieve_analogs(model, store, train_pos, target_timestamps, device, k=K,
                     train_emb=None):
    """K=15 nearest training analogs per target in embedding space.

    Returns (members, analog_positions, distances): members (n_targets, K)
    observed gusts, analog_positions (n_targets, K) archive positions,
    distances (n_targets, K) embedding distances. Applies the +/-168 h
    exclusion around every target; pool is training only. Pass a
    precomputed train_emb to avoid re-embedding the archive per target.
    """
    if train_emb is None:
        train_emb = embed_positions(model, store, train_pos, device)
    train_time = store.df["timestamp"].to_numpy(dtype="datetime64[ns]")[train_pos]
    members, analog_pos, dists_out = [], [], []
    for ts in target_timestamps:
        row = store.df[store.df["timestamp"] == ts]
        assert len(row) == 1, f"target {ts} not in archive"
        i = row.index[0]
        x = torch.tensor(store.seqs[i:i + 1], device=device)
        with torch.no_grad():
            z = model(x).cpu().numpy()[0]
        dt_h = np.abs((train_time - _to_naive_np_datetime64(ts)).astype("timedelta64[h]").astype(float))
        eligible = np.where(dt_h > EXCLUSION_HOURS)[0]
        dists = np.linalg.norm(train_emb[eligible] - z, axis=1)
        order = np.argsort(dists)[:k]
        chosen = train_pos[eligible[order]]
        members.append(store.df.loc[chosen, TARGET].to_numpy(dtype=float))
        analog_pos.append(chosen)
        dists_out.append(dists[order])
    return np.array(members), np.array(analog_pos), np.array(dists_out)


def evaluate_phase4(model, store, train_pos, test_df, device,
                    n_events_expected=None):
    """Evaluate on the shared storm events with strict assertions.

    n_events_expected is asserted only when given (130 for the 1-year
    regime); otherwise the actual event count is used. Returns
    (metrics_dict, members, raw_fcst, obs).
    """
    storms = get_storm_events(test_df)
    event_ts = list(storms["timestamp"])
    # Strict test protocol (assertions from the task spec).
    if n_events_expected is not None:
        assert len(event_ts) == n_events_expected, \
            f"expected {n_events_expected} events, got {len(event_ts)}"
    assert len(set(event_ts)) == len(event_ts), "event timestamps are not unique"
    members, analog_pos, _ = retrieve_analogs(model, store, train_pos, event_ts, device)
    assert members.shape == (len(event_ts), K), f"bad ensemble shape {members.shape}"
    obs = storms[TARGET].to_numpy(dtype=float)
    raw = storms["fcst_gust"].to_numpy(dtype=float)
    # No analog within +/-168 h; no analog outside the training archive.
    train_set = set(train_pos.tolist())
    for ts, apos in zip(event_ts, analog_pos):
        assert all(p in train_set for p in apos), "analog outside training archive"
        gaps = np.abs((store.df["timestamp"].to_numpy(dtype="datetime64[ns]")[apos]
                       - _to_naive_np_datetime64(ts)).astype("timedelta64[h]").astype(float))
        assert gaps.min() > EXCLUSION_HOURS, "analog inside exclusion window"
    assert np.isfinite(members).all() and np.isfinite(obs).all()

    rmse = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))
    ens_mean = members.mean(axis=1)
    rmse_raw = rmse(raw, obs)
    rmse_triplet = rmse(ens_mean, obs)
    crps_raw = float(np.mean(np.abs(raw - obs)))  # single-member CRPS = MAE
    crps_triplet = float(np.mean(properscoring.crps_ensemble(obs, members)))
    return {
        "rmse_raw_nwp_kmh": round(rmse_raw, 2),
        "rmse_triplet_kmh": round(rmse_triplet, 2),
        "rmse_skill_pct": round(100 * (rmse_raw - rmse_triplet) / rmse_raw, 1),
        "crps_raw_nwp_kmh": round(crps_raw, 2),
        "crps_triplet_kmh": round(crps_triplet, 2),
        "crps_skill_pct": round(100 * (crps_raw - crps_triplet) / crps_raw, 1),
        "n_test_storm_events": len(event_ts),
    }, members, raw, obs


# ---------------------------------------------------------------------------
# Checkpoint helpers (for the comparison workflow).
# ---------------------------------------------------------------------------
def save_checkpoint(model, history, path, scaler, regime=None):
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": {
            "input_features": 4, "hidden_features": HIDDEN,
            "hidden_layers": LAYERS, "output_features": EMBED_DIM,
            "dropout": DROPOUT, "seq_len": SEQ_LEN, "lr": LR,
            "margin": MARGIN, "patience": PATIENCE,
            "max_epochs": MAX_EPOCHS, "batch": BATCH,
            "num_analogs": NUM_ANALOGS,
            "fitness_num_negative": FITNESS_NUM_NEGATIVE,
            "anchors_per_epoch": ANCHORS_PER_EPOCH, "seed": SEED,
            "regime": regime,
        },
        "history": {k: v for k, v in history.items()},
        "feat_scaler_mean": scaler.mean_.tolist(),
        "feat_scaler_scale": scaler.scale_.tolist(),
    }, path)


def load_triplet_model(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = DeepAnalogLSTM(**{k: ckpt["config"][k] for k in
                              ("input_features", "hidden_features",
                               "hidden_layers", "output_features", "dropout")})
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model, ckpt


def main():
    parser = argparse.ArgumentParser(description="Phase 4: Hu et al. Deep Analog")
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--anchors-per-epoch", type=int, default=ANCHORS_PER_EPOCH)
    parser.add_argument("--val-anchors-max", type=int, default=500,
                        help="max validation anchors for the fixed val "
                             "triplet set (default 500 -> 7,500 triplets)")
    parser.add_argument("--skip-train", action="store_true",
                        help="load the saved checkpoint and evaluate only")
    parser.add_argument("--resume", action="store_true",
                        help="if the checkpoint already exists, restore its "
                             "weights/history and continue training from the "
                             "next epoch (same regime/config; optimizer and "
                             "scheduler state start fresh)")
    parser.add_argument("--archive", type=str, default=None,
                        help="archive CSV (default: 1-year file; regime "
                             "branches by filename)")
    parser.add_argument("--regime", type=str, default=None,
                        help="explicit regime (e.g. nested_1yr); loads the "
                             "long archive yearly files for nested regimes")
    parser.add_argument("--ckpt-out", type=str, default=None,
                        help="checkpoint .pth path (default: "
                             "models/deep_anen_triplet_weights.pth)")
    parser.add_argument("--weights-json-out", type=str, default=None,
                        help="text-safe weights JSON path")
    parser.add_argument("--metrics-out", type=str, default=None,
                        help="metrics JSON path (default: "
                             "metrics/deep_anen_triplet_metrics.json)")
    parser.add_argument("--errors-out", type=str, default=None,
                        help="optional .npz path for per-event arrays")
    args = parser.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"PyTorch {torch.__version__} | device: {device}")

    if args.regime is not None:
        from data_prep import NESTED_EXPECTED_ROWS, NESTED_REGIMES
        df, train_df, val_df, test_df, regime = load_and_split(regime=args.regime)
        print(f"Archive: regime {args.regime} | regime: {regime}")
    else:
        archive_path = args.archive or os.path.join(here, "data",
                                                    "atlanta_weather_archive_1yr.csv")
        df, train_df, val_df, test_df, regime = load_and_split(archive_path)
        print(f"Archive: {archive_path} | regime: {regime}")
    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    if regime == "1yr":
        assert len(train_df) == 6888 and len(val_df) == 384 and len(test_df) == 1488
        n_events_expected = 130
    elif regime in NESTED_REGIMES:
        exp = NESTED_EXPECTED_ROWS[regime]
        assert (len(train_df), len(val_df), len(test_df)) == exp, \
            f"{regime}: expected splits {exp}, got " \
            f"({len(train_df)}, {len(val_df)}, {len(test_df)})"
        n_events_expected = None
    else:
        assert len(train_df) == 59928 and len(val_df) == 8760 and len(test_df) == 8760
        n_events_expected = None

    scaler = fit_feature_scaler(train_df)
    store = SequenceStore(df, scaler)
    train_pos = store.eligible_positions(train_df)
    val_pos = store.eligible_positions(val_df)
    print(f"Eligible training positions (full {SEQ_LEN}h lookback): {len(train_pos)}")
    print(f"Eligible validation positions: {len(val_pos)}")
    obs = df[TARGET].to_numpy(dtype=float)
    timestamps = df["timestamp"].to_numpy(dtype="datetime64[ns]")

    ckpt_path = args.ckpt_out or os.path.join(here, "models",
                                              "deep_anen_triplet_weights.pth")
    weights_json_path = args.weights_json_out or os.path.join(
        here, "models", "deep_anen_triplet_weights.json")
    os.makedirs(os.path.join(here, "models"), exist_ok=True)

    if args.skip_train:
        model, ckpt = load_triplet_model(ckpt_path, device)
        print(f"Loaded checkpoint (best epoch {ckpt['history']['best_epoch']})")
        history = ckpt["history"]
    else:
        resume_from = ckpt_path if (args.resume and os.path.exists(ckpt_path)) else None
        model, history = train_triplet_model(
            store, train_pos, val_pos, obs, timestamps, device,
            max_epochs=args.epochs, anchors_per_epoch=args.anchors_per_epoch,
            ckpt_path=ckpt_path, scaler=scaler,
            val_anchors_max=args.val_anchors_max, regime=regime,
            resume_from=resume_from)
        save_checkpoint(model, history, ckpt_path, scaler, regime=regime)
        # Text-safe copy of the weights (GitHub connector is text-only).
        weights_json = {k: v.cpu().tolist()
                        for k, v in model.state_dict().items()}
        with open(weights_json_path, "w") as f:
            json.dump({"config": {"input_features": 4, "hidden_features": HIDDEN,
                                  "hidden_layers": LAYERS, "output_features": EMBED_DIM,
                                  "dropout": DROPOUT, "seq_len": SEQ_LEN},
                       "state_dict": weights_json}, f)
        print(f"Saved {ckpt_path}")

    metrics, members, raw, obs_ev = evaluate_phase4(
        model, store, train_pos, test_df, device,
        n_events_expected=n_events_expected)
    metrics_full = {
        "model": "Deep Analog (Hu et al., 2023) — LSTM triplet embedding",
        "paper": PAPER,
        "paper_repo": PAPER_REPO,
        "method_note": "LSTM (3 layers, 20 hidden, dropout 0.01) + linear layer to 20-D embeddings; TripletMarginLoss margin 0.9 (official default); Reverse Analog fitness-proportionate sampling; 24 h lookback sequences (adaptation: paper uses lead-time windows). No L2 normalization (verified in official code). No regression head.",
        "regime": regime,
        "n_train_rows": int(len(train_df)),
        "rmse_kmh": metrics["rmse_triplet_kmh"],
        "crps_kmh": metrics["crps_triplet_kmh"],
        "rmse_raw_nwp_kmh": metrics["rmse_raw_nwp_kmh"],
        "crps_raw_nwp_kmh": metrics["crps_raw_nwp_kmh"],
        "rmse_skill_vs_raw_pct": metrics["rmse_skill_pct"],
        "crps_skill_vs_raw_pct": metrics["crps_skill_pct"],
        "n_test_storm_events": metrics["n_test_storm_events"],
        "K": K,
        "exclusion_window_hours": EXCLUSION_HOURS,
        "sequence_length_hours": SEQ_LEN,
        "training": {
            "optimizer": "Adam", "lr": LR, "margin": MARGIN,
            "batch": BATCH, "patience": PATIENCE,
            "max_epochs": args.epochs,
            "epochs_trained": history["epochs_trained"],
            "best_epoch": history["best_epoch"],
            "best_val_loss": round(history["best_val_loss"], 4),
            "early_stopped": history["early_stopped"],
            "anchors_per_epoch": args.anchors_per_epoch,
            "val_anchors_max": args.val_anchors_max,
            "num_analogs_positive_group": NUM_ANALOGS,
            "fitness_num_negative": FITNESS_NUM_NEGATIVE,
            "train_loss_final": round(history["train_loss"][-1], 4),
            "val_loss_final": round(history["val_loss"][-1], 4),
        },
        "random_seed": SEED,
    }
    out_path = args.metrics_out or os.path.join(here, "metrics",
                                                "deep_anen_triplet_metrics.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(metrics_full, f, indent=2)

    if args.errors_out:
        storms = get_storm_events(test_df)
        np.savez(args.errors_out,
                 timestamps=np.array([str(t) for t in storms["timestamp"]]),
                 raw_fcst=raw, obs=obs_ev, members=members)
        print(f"Saved per-event arrays to {args.errors_out}")

    table = pd.DataFrame({
        "Method": ["Raw NWP", "Deep Analog (Hu et al., 2023)"],
        "RMSE (km/h)": [metrics["rmse_raw_nwp_kmh"], metrics["rmse_triplet_kmh"]],
        "CRPS (km/h)": [metrics["crps_raw_nwp_kmh"], metrics["crps_triplet_kmh"]],
    })
    print("\n" + table.to_string(index=False))
    print(f"\nRMSE skill: {metrics['rmse_skill_pct']:+.1f}% | "
          f"CRPS skill: {metrics['crps_skill_pct']:+.1f}%")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
