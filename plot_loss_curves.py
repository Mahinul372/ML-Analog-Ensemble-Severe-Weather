"""Plot training vs validation loss curves for both neural models, both regimes.

Overfitting evidence for the journal submission (mandatory). Sources:

  * Deep-AnEn-Regression, 1-year: per-epoch train/val MSE saved in the
    Phase 2 checkpoint (models/deep_anen_regression_weights.pth,
    "history" key). Recovered — no retraining.
  * Deep-AnEn-Regression, long archive: per-epoch train/val MSE saved in
    the long checkpoint (models/deep_anen_regression_weights_10yr.pth).
  * Deep Analog triplet, 1-year: per-epoch train/val triplet loss parsed
    from ~/workspace/phase4_train.log (100 epochs). Recovered — no
    retraining.
  * Deep Analog triplet, long archive: per-epoch train/val triplet loss
    from the long checkpoint (models/deep_anen_triplet_weights_10yr.pth),
    saved on every validation improvement during training. Available
    only after the long triplet run finishes.

Traditional AnEn has NO loss curve: it is non-parametric (no learned
parameters, nothing to overfit in the neural sense). No curve is
fabricated for it; the notebook and README document this explicitly.

Outputs (per curve):
  plots/train_val_loss_<method>_<regime>.png  (300 dpi, local-git only)
  plots/train_val_loss_<method>_<regime>.svg  (text-safe, on GitHub)
  metrics/train_val_loss_<method>_<regime>.json (underlying history)

Usage:
    python plot_loss_curves.py            # generate every available curve
    python plot_loss_curves.py --only triplet_10yr
"""

import argparse
import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
WORKSPACE = os.path.dirname(HERE)


# ---------------------------------------------------------------------------
# History sources
# ---------------------------------------------------------------------------
def _history_from_pth(path):
    """Per-epoch train/val lists from a checkpoint's saved history dict.

    Regression checkpoints store the lists under "train"/"val";
    triplet checkpoints under "train_loss"/"val_loss". Both are accepted.
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    h = ckpt.get("history")
    if not h:
        raise ValueError(f"no per-epoch history in {path}")
    train = h.get("train", h.get("train_loss"))
    val = h.get("val", h.get("val_loss"))
    if train is None or val is None:
        raise ValueError(f"no per-epoch train/val history in {path}")
    val = [float(x) for x in val]
    best_i = int(np.argmin(val)) if val else None
    return {"train": [float(x) for x in train],
            "val": val,
            "best_epoch": h.get("best_epoch", best_i + 1 if best_i is not None else None),
            "best_val_loss": h.get("best_val_loss",
                                   val[best_i] if best_i is not None else None)}


def _triplet_history_from_log(path):
    """Parse 'Epoch N/M | train X | val Y | best val Z' lines."""
    pat = re.compile(
        r"Epoch\s+(\d+)/(\d+) \| train ([\d.]+) \| val ([\d.]+) \| best val ([\d.]+)")
    ep, tr, va = [], [], []
    with open(path) as f:
        for line in f:
            m = pat.search(line)
            if m:
                ep.append(int(m.group(1)))
                tr.append(float(m.group(3)))
                va.append(float(m.group(4)))
    if not ep:
        raise ValueError(f"no epoch lines found in {path}")
    best_i = int(np.argmin(va))
    return {"train": tr, "val": va,
            "best_epoch": ep[best_i], "best_val_loss": va[best_i]}


CURVES = {
    # key: (method label, regime label, loss label, source callable)
    "regression_1yr": (
        "Deep-AnEn-Regression", "1-year", "MSE (standardized gust)",
        lambda: _history_from_pth(os.path.join(
            HERE, "models", "deep_anen_regression_weights.pth"))),
    "regression_10yr": (
        "Deep-AnEn-Regression", "long archive", "MSE (standardized gust)",
        lambda: _history_from_pth(os.path.join(
            HERE, "models", "deep_anen_regression_weights_10yr.pth"))),
    "triplet_1yr": (
        "Deep Analog (Hu et al., 2023)", "1-year", "Triplet margin loss",
        lambda: _triplet_history_from_log(os.path.join(
            WORKSPACE, "phase4_train.log"))),
    "triplet_10yr": (
        "Deep Analog (Hu et al., 2023)", "long archive", "Triplet margin loss",
        lambda: _history_from_pth(os.path.join(
            HERE, "models", "deep_anen_triplet_weights_10yr.pth"))),
}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_curve(key, hist, method, regime, loss_label):
    epochs = np.arange(1, len(hist["train"]) + 1)
    train = np.array(hist["train"])
    val = np.array(hist["val"])
    best_ep = hist.get("best_epoch")
    best_val = hist.get("best_val_loss")

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(epochs, train, color="#2c7fb8", lw=2, label="Training loss")
    ax.plot(epochs, val, color="#d7301f", lw=2, label="Validation loss")
    if best_ep is not None:
        ax.axvline(best_ep, color="#636363", lw=1, ls="--",
                   label=f"Best validation epoch {best_ep}")
        ax.plot(best_ep, best_val, "o", color="#636363", ms=6)
    ax.set_title(f"{method} — training vs validation loss ({regime} regime)\n"
                 f"{loss_label}; best val {best_val:.4f} at epoch {best_ep} "
                 f"of {len(epochs)}",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(loss_label)
    ax.set_xlim(1, len(epochs))
    ax.legend(loc="upper right", framealpha=0.9)
    fig.tight_layout()

    base = os.path.join(HERE, "plots", f"train_val_loss_{key}")
    fig.savefig(base + ".png", dpi=300)
    fig.savefig(base + ".svg")
    plt.close(fig)

    # Underlying history (JSON): the exact numbers behind the figure.
    out = {
        "method": method,
        "regime": regime,
        "loss": loss_label,
        "epochs": [int(e) for e in epochs],
        "train_loss": [float(x) for x in train],
        "val_loss": [float(x) for x in val],
        "best_epoch": int(best_ep) if best_ep is not None else None,
        "best_val_loss": float(best_val) if best_val is not None else None,
    }
    with open(os.path.join(HERE, "metrics",
                           f"train_val_loss_{key}.json"), "w") as f:
        json.dump(out, f, indent=2)
    gap = float(val[-1] - train[-1])
    print(f"{key}: {len(epochs)} epochs | train {train[0]:.4f} -> {train[-1]:.4f} | "
          f"val {val[0]:.4f} -> {val[-1]:.4f} | best val {best_val:.4f} (ep {best_ep}) | "
          f"final train-val gap {gap:+.4f}")
    print(f"  saved plots/train_val_loss_{key}.png/.svg + "
          f"metrics/train_val_loss_{key}.json")


def main():
    parser = argparse.ArgumentParser(description="Plot train/val loss curves")
    parser.add_argument("--only", type=str, default=None,
                        help="comma-separated curve keys to generate "
                             "(default: all available)")
    args = parser.parse_args()
    wanted = (args.only.split(",") if args.only
              else list(CURVES.keys()))
    os.makedirs(os.path.join(HERE, "plots"), exist_ok=True)
    os.makedirs(os.path.join(HERE, "metrics"), exist_ok=True)
    done, missing = [], []
    for key in wanted:
        if key not in CURVES:
            print(f"Unknown curve key: {key} (expected one of {list(CURVES)})")
            continue
        method, regime, loss_label, source = CURVES[key]
        try:
            hist = source()
        except Exception as exc:  # noqa: BLE001 - report and continue
            print(f"{key}: NOT AVAILABLE ({exc})")
            missing.append(key)
            continue
        plot_curve(key, hist, method, regime, loss_label)
        done.append(key)
    print(f"\nGenerated: {done}" + (f" | Missing: {missing}" if missing else ""))


if __name__ == "__main__":
    main()
