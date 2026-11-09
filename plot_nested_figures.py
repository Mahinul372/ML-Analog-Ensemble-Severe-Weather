"""Figures 1-4 for the nested-archive revision (Phase B).

Fig 1 — Experimental framework diagram (matplotlib boxes/arrows).
Fig 2 — Overall forecast verification, nested_8yr as the primary
  configuration: (a) RMSE/MAE/CRPS for all five methods; (b) score
  differences vs Traditional AnEn with paired block-bootstrap 95% CIs;
  (c) calibration: rank (PIT) histograms + 90% interval coverage for
  the three ensemble methods.
Fig 3 — Archive-size sensitivity (the central result): RMSE and CRPS
  vs 1/3/5/8-year archives for the three analog methods, MOS and Raw
  NWP as reference lines.
Fig 4 — Representative high-gust episode (predefined rule: highest
  observed peak gust in the test period): observations, Raw NWP, MOS,
  and the three ensemble forecasts (mean + 90% prediction interval).

Inputs: metrics/nested_comparison_metrics.json,
metrics/nested_episode_forecasts.json.
Outputs: plots/nested_fig{1,2,3,4}_*.png and .svg (PNG local-git only,
SVG pushed to GitHub).

Usage: python plot_nested_figures.py
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
PLOTS = os.path.join(HERE, "plots")
METRICS = os.path.join(HERE, "metrics")
os.makedirs(PLOTS, exist_ok=True)

METHODS = ["Raw NWP", "MOS", "Traditional AnEn",
           "Deep-AnEn-Regression", "Deep Analog (Hu et al., 2023)"]
SHORT = {"Raw NWP": "Raw NWP", "MOS": "MOS",
         "Traditional AnEn": "Traditional\nAnEn",
         "Deep-AnEn-Regression": "Deep-AnEn-\nRegression",
         "Deep Analog (Hu et al., 2023)": "Deep Analog\n(Hu et al., 2023)"}
COLORS = {"Raw NWP": "#7f7f7f", "MOS": "#9467bd",
          "Traditional AnEn": "#1f77b4",
          "Deep-AnEn-Regression": "#ff7f0e",
          "Deep Analog (Hu et al., 2023)": "#2ca02c"}
ARCHIVES = ["nested_1yr", "nested_3yr", "nested_5yr", "nested_8yr"]
ARCH_LABELS = ["1-year", "3-year", "5-year", "8-year"]
ARCH_X = [1, 3, 5, 8]


def save(fig, name, tight=True):
    if tight:
        fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, name + ".png"), dpi=200)
    fig.savefig(os.path.join(PLOTS, name + ".svg"))
    plt.close(fig)
    print(f"Saved plots/{name}.png/.svg")


# ---------------------------------------------------------------------------
# Fig 1 — framework diagram
# ---------------------------------------------------------------------------
def fig1():
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis("off")

    def box(x, y, w, h, text, fc="#eef3f8", ec="#1f77b4", fs=9.5, weight="normal"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                                    fc=fc, ec=ec, lw=1.2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, weight=weight, wrap=True)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                                     arrowstyle="-|>", mutation_scale=14,
                                     color="#333333", lw=1.2))

    ax.text(6, 6.75, "Experimental framework — nested-archive design",
            ha="center", fontsize=13, weight="bold")
    # Data sources.
    box(0.3, 5.5, 5.4, 0.85,
        "Open-Meteo Previous-Runs API\nNWP forecasts (provider best-match:\nNOAA GFS, Day-0, 0–6 h lead)",
        fc="#fff7e6", ec="#ff7f0e")
    box(6.3, 5.5, 5.4, 0.85,
        "Open-Meteo Archive API\nERA5 reanalysis reference data\n(not station observations)",
        fc="#e6f5e6", ec="#2ca02c")
    arrow(3.0, 5.5, 4.6, 4.95)
    arrow(9.0, 5.5, 7.4, 4.95)
    box(2.6, 4.15, 6.8, 0.8,
        "Atlanta, GA hourly archive  2018-01-01 → 2026-11-01\n77,448 rows, 0 missing values, 0 gaps",
        weight="bold")
    arrow(6.0, 4.15, 6.0, 3.85)
    # Splits.
    box(0.3, 2.85, 3.7, 0.95,
        "TRAINING — ends 2025-09-30 23:00\nstart varies: 2024-11-01 (1 yr) /\n2022-11-01 (3 yr) / 2020-11-01 (5 yr) /\n2018-01-01 (8 yr, data available from source)",
        fs=8.5)
    box(4.2, 2.85, 3.6, 0.95,
        "VALIDATION — 2025-10-01 → 2025-10-31\n(744 h) early stopping and\nmodel selection ONLY",
        fs=8.5)
    box(8.0, 2.85, 3.7, 0.95,
        "TEST — 2025-11-01 → 2026-11-01 (fixed)\n8,784 h; 1,286 high-gust events\n(obs ≥ 35 km/h); 81 episodes",
        fs=8.5)
    arrow(2.15, 2.85, 2.15, 2.5)
    arrow(6.0, 2.85, 6.0, 2.5)
    arrow(9.85, 2.85, 9.85, 2.5)
    # Methods.
    box(0.3, 1.35, 2.1, 1.0, "Raw NWP\n(deterministic)", fc="#f2f2f2", ec="#7f7f7f", fs=9)
    box(2.6, 1.35, 2.1, 1.0, "MOS\nOLS regression", fc="#f3eefc", ec="#9467bd", fs=9)
    box(4.9, 1.35, 2.1, 1.0, "Traditional AnEn\n4-D analogs", fc="#e6f0fa", ec="#1f77b4", fs=9)
    box(7.2, 1.35, 2.1, 1.0, "Deep-AnEn-\nRegression 16-D", fc="#fff0e0", ec="#ff7f0e", fs=9)
    box(9.5, 1.35, 2.1, 1.0, "Deep Analog\n(Hu et al., 2023) 20-D", fc="#e6f5e6", ec="#2ca02c", fs=9)
    ax.text(6, 2.62, "Five methods — identical protocol: K = 15, ±168 h exclusion, analogs from the training pool only",
            ha="center", fontsize=9.5, style="italic")
    arrow(6.0, 1.35, 6.0, 1.05)
    box(1.2, 0.15, 9.6, 0.85,
        "Verification: RMSE / MAE / CRPS  •  calibration (rank histogram, 90% interval coverage, spread–skill)\n"
        "Uncertainty: paired block-bootstrap 95% CIs (episode blocks) + Diebold–Mariano tests",
        fc="#fdf2e9", ec="#d35400", fs=9.5)
    save(fig, "nested_fig1_framework")


# ---------------------------------------------------------------------------
# Fig 2 — 8-year verification + bootstrap CIs + calibration
# ---------------------------------------------------------------------------
def fig2(comp):
    reg = comp["regimes"]["nested_8yr"]
    scores = reg["scores"]
    boot = reg["block_bootstrap"]["vs_Traditional AnEn"]
    calib = reg["calibration"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Overall forecast verification — 8-year archive (primary configuration)\n"
                 "Test: 2025-11-01 → 2026-11-01, 1,286 high-gust events, 81 episodes",
                 fontsize=12, weight="bold")
    # (a) Absolute scores.
    ax = axes[0, 0]
    x = np.arange(len(METHODS))
    w = 0.25
    for j, key in enumerate(["rmse_kmh", "mae_kmh", "crps_kmh"]):
        vals = [scores[m][key] for m in METHODS]
        ax.bar(x + (j - 1) * w, vals, w, label=key.replace("_kmh", "").upper())
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT[m] for m in METHODS], fontsize=8)
    ax.set_ylabel("km/h")
    ax.set_title("(a) Scores on the 8-year test period", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    # (b) Bootstrap differences vs Traditional AnEn.
    ax = axes[0, 1]
    others = [m for m in METHODS if m != "Traditional AnEn"]
    x = np.arange(len(others))
    w = 0.35
    for j, key in enumerate(["rmse_diff_kmh", "crps_diff_kmh"]):
        means = [boot[m][key]["mean"] for m in others]
        lo = [boot[m][key]["mean"] - boot[m][key]["ci_lower"] for m in others]
        hi = [boot[m][key]["ci_upper"] - boot[m][key]["mean"] for m in others]
        ax.bar(x + (j - 0.5) * w, means, w,
               yerr=[lo, hi], capsize=4,
               label="Δ" + key.split("_")[0].upper())
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT[m] for m in others], fontsize=8)
    ax.set_ylabel("Score difference vs Traditional AnEn (km/h)")
    ax.set_title("(b) Paired block-bootstrap 95% CIs (episode blocks, B=2000)",
                 fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    # (c) Rank histograms for the three ensemble methods (bottom-left).
    ens = ["Traditional AnEn", "Deep-AnEn-Regression", "Deep Analog (Hu et al., 2023)"]
    ax = axes[1, 0]
    counts = np.array([calib[m]["rank_histogram_counts"] for m in ens], dtype=float)
    counts /= counts.sum(axis=1, keepdims=True)
    xx = np.arange(16)
    for i, m in enumerate(ens):
        ax.plot(xx, counts[i], marker="o", ms=3, label=SHORT[m].replace("\n", " "))
    ax.axhline(1 / 16, color="k", ls="--", lw=0.9, label="uniform")
    ax.set_xlabel("Rank of observation among 15 members")
    ax.set_ylabel("Fraction")
    ax.set_title("(c) Rank (PIT) histograms — 8-year archive", fontsize=10)
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.3)
    ax = axes[1, 1]
    cov = [calib[m]["pi90_coverage"] for m in ens]
    ssr = [calib[m]["spread_skill_ratio"] for m in ens]
    xx = np.arange(len(ens))
    ax.bar(xx - 0.2, cov, 0.4, label="90% interval coverage (nominal 0.90)")
    ax.bar(xx + 0.2, ssr, 0.4, label="Spread–skill ratio (ideal 1.0)")
    ax.axhline(0.90, color="C0", ls="--", lw=0.9)
    ax.axhline(1.0, color="C1", ls="--", lw=0.9)
    ax.set_xticks(xx)
    ax.set_xticklabels([SHORT[m] for m in ens], fontsize=8)
    ax.set_ylim(0, 1.25)
    ax.set_title("(d) Calibration summary — 8-year archive", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    save(fig, "nested_fig2_verification_8yr")


# ---------------------------------------------------------------------------
# Fig 3 — archive-size sensitivity
# ---------------------------------------------------------------------------
def fig3(comp):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Archive-size sensitivity — fixed test period 2025-11-01 → 2026-11-01\n"
                 "1,286 high-gust events; only the training length varies",
                 fontsize=12, weight="bold")
    ens = ["Traditional AnEn", "Deep-AnEn-Regression", "Deep Analog (Hu et al., 2023)"]
    for ax, key, title in [(axes[0], "rmse_kmh", "(a) RMSE vs archive size"),
                           (axes[1], "crps_kmh", "(b) CRPS vs archive size")]:
        for m in ens:
            vals = [comp["regimes"][r]["scores"][m][key] for r in ARCHIVES]
            ax.plot(ARCH_X, vals, marker="o", label=SHORT[m].replace("\n", " "),
                    color=COLORS[m])
        for m, ls in [("MOS", "--"), ("Raw NWP", ":")]:
            vals = [comp["regimes"][r]["scores"][m][key] for r in ARCHIVES]
            ax.plot(ARCH_X, vals, ls=ls, marker="s", ms=4, label=m, color=COLORS[m])
        ax.set_xticks(ARCH_X)
        ax.set_xticklabels(ARCH_LABELS)
        ax.set_xlabel("Training archive length")
        ax.set_ylabel("km/h")
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(alpha=0.3)
    save(fig, "nested_fig3_archive_sensitivity")


# ---------------------------------------------------------------------------
# Fig 4 — representative episode
# ---------------------------------------------------------------------------
def fig4(ep):
    import pandas as pd
    times = pd.to_datetime(ep["timestamps"])
    obs = np.array(ep["obs_gust"])
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(times, obs, "k-", lw=2, label="ERA5 reference (observed gust)")
    ax.plot(times, np.array(ep["raw_nwp_gust"]), color=COLORS["Raw NWP"],
            lw=1.5, label="Raw NWP")
    ax.plot(times, np.array(ep["mos_gust"]), color=COLORS["MOS"],
            lw=1.5, label="MOS")
    for key, label in [("traditional_anen", "Traditional AnEn"),
                       ("deep_anen_regression", "Deep-AnEn-Regression"),
                       ("deep_analog", "Deep Analog (Hu et al., 2023)")]:
        d = ep[key]
        mean = np.array(d["mean"])
        ax.plot(times, mean, color=COLORS[label], lw=1.5, label=label + " mean")
        ax.fill_between(times, np.array(d["p5"]), np.array(d["p95"]),
                        color=COLORS[label], alpha=0.15, label=label + " 90% interval")
    ax.axhline(35, color="r", ls="--", lw=1, label="High-gust threshold (35 km/h)")
    ax.axvspan(pd.to_datetime(ep["episode_start"]), pd.to_datetime(ep["episode_end"]),
               color="red", alpha=0.06)
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Wind gust (km/h)")
    ax.set_title(f"Representative high-gust episode — peak {ep['peak_gust_kmh']:.1f} km/h "
                 f"at {ep['peak_timestamp']}\n"
                 f"Episode {ep['episode_start']} → {ep['episode_end']} "
                 f"(±24 h window shown); 8-year archive",
                 fontsize=11)
    # Legend below the axes so it never covers the episode peak.
    ax.legend(fontsize=8, ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, -0.26))
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.subplots_adjust(bottom=0.36, top=0.86)
    save(fig, "nested_fig4_episode", tight=False)


def main():
    with open(os.path.join(METRICS, "nested_comparison_metrics.json")) as f:
        comp = json.load(f)
    with open(os.path.join(METRICS, "nested_episode_forecasts.json")) as f:
        ep = json.load(f)
    fig1()
    fig2(comp)
    fig3(comp)
    fig4(ep)


if __name__ == "__main__":
    main()
