"""Nested-archive comparison analysis — Phase B.

Loads the per-event arrays saved by the five method runs on each of
the four nested archives and produces the final verification table:

  * RMSE / MAE / CRPS per method per archive (cross-checked against
    the saved per-method metrics JSONs).
  * Calibration for the three ensemble methods (Traditional AnEn,
    Deep-AnEn-Regression, Deep Analog): rank (PIT) histogram counts,
    90% prediction-interval coverage, spread-skill ratio.
  * Uncertainty (reviewer revision 6): PAIRED BLOCK-BOOTSTRAP 95%
    confidence intervals for the performance DIFFERENCE of each method
    vs Traditional AnEn and vs Raw NWP. Blocks are the 81 independent
    episodes of the fixed test period (declustering gap >= 24 h; see
    data_prep.get_episodes): whole episodes are resampled with
    replacement (B=2000, seed 42), which preserves the temporal
    dependence between high-gust events of the same storm. Percentile
    CIs on the resampled score differences.
  * Diebold-Mariano tests (same variant as dm_tests.py: loss
    differential d = |e_method| - |e_reference| on per-event absolute
    errors, Newey-West HAC Bartlett kernel, lag h = floor(T^(1/3)),
    two-sided normal p), vs Traditional AnEn and vs Raw NWP.

Inputs (all in metrics/):
  nested_mos_errors_<regime>.npz          (timestamps, raw_fcst, obs, mos)
  nested_traditional_errors_<regime>.npz  (timestamps, raw_fcst, obs, members)
  nested_regression_errors_<regime>.npz   (timestamps, raw_fcst, obs, members)
  nested_triplet_errors_<regime>.npz      (timestamps, raw_fcst, obs, members)

Outputs:
  metrics/nested_comparison_metrics.json   (scores + calibration + DM)
  metrics/nested_bootstrap_cis.json        (block-bootstrap 95% CIs)
  metrics/nested_table3.csv                (Table 3: scores + uncertainty)

Usage: python nested_analysis.py
"""

import json
import os

import numpy as np
import pandas as pd
import properscoring

from data_prep import (
    NESTED_REGIMES,
    STORM_THRESHOLD,
    assign_episode_ids,
    get_storm_events,
    load_and_split,
)
from dm_tests import diebold_mariano

HERE = os.path.dirname(os.path.abspath(__file__))
METRICS = os.path.join(HERE, "metrics")
BOOTSTRAP_B = 2000
BOOTSTRAP_SEED = 42

METHODS = ["Raw NWP", "MOS", "Traditional AnEn",
           "Deep-AnEn-Regression", "Deep Analog (Hu et al., 2023)"]


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def mae(a, b):
    return float(np.mean(np.abs(a - b)))


def crps_ens(obs, members):
    return float(np.mean(properscoring.crps_ensemble(obs, members)))


def calibration(obs, members):
    """Rank (PIT) histogram counts, 90% PI coverage, spread-skill ratio.

    Ranks: position of the observation among the 15 sorted members
    (16 possible ranks, 0..15). 90% interval = p5/p95 of the members.
    Spread-skill ratio = mean ensemble std / RMSE of the ensemble mean
    (1.0 = perfectly calibrated spread on average).
    """
    n, k = members.shape
    ranks = np.array([int((members[i] < obs[i]).sum()) for i in range(n)])
    rank_counts = [int((ranks == r).sum()) for r in range(k + 1)]
    lo = np.percentile(members, 5, axis=1)
    hi = np.percentile(members, 95, axis=1)
    coverage = float(np.mean((obs >= lo) & (obs <= hi)))
    spread = float(np.mean(members.std(axis=1, ddof=1)))
    skill = rmse(members.mean(axis=1), obs)
    return {
        "rank_histogram_counts": rank_counts,
        "rank_histogram_n_bins": k + 1,
        "pi90_coverage": round(coverage, 3),
        "pi90_coverage_nominal": 0.90,
        "spread_kmh": round(spread, 2),
        "spread_skill_ratio": round(spread / skill, 3),
    }


def block_bootstrap_ci(diff_fn, ep_ids, n_episodes, rng, B=BOOTSTRAP_B):
    """Percentile 95% CI of a score-difference statistic under episode
    block resampling. diff_fn(idx) returns the statistic on resampled
    event indices idx (whole episodes drawn with replacement)."""
    stats = []
    for _ in range(B):
        sampled_eps = rng.integers(0, n_episodes, size=n_episodes)
        idx = np.concatenate([np.where(ep_ids == e)[0] for e in sampled_eps])
        stats.append(diff_fn(idx))
    stats = np.array(stats)
    return {
        "mean": round(float(stats.mean()), 3),
        "ci_lower": round(float(np.percentile(stats, 2.5)), 3),
        "ci_upper": round(float(np.percentile(stats, 97.5)), 3),
        "n_bootstrap": B,
    }


def analyse_regime(regime):
    # Episode ids are identical for every archive (fixed test period).
    _, _, _, test_df, _ = load_and_split(regime=regime)
    events = get_storm_events(test_df, STORM_THRESHOLD)
    ep_ids = assign_episode_ids(events)
    n_episodes = int(ep_ids.max()) + 1
    assert len(events) == 1286 and n_episodes == 81, \
        f"{regime}: expected 1286 events / 81 episodes, got {len(events)} / {n_episodes}"

    mos = np.load(os.path.join(METRICS, f"nested_mos_errors_{regime}.npz"))
    trad = np.load(os.path.join(METRICS, f"nested_traditional_errors_{regime}.npz"))
    regr = np.load(os.path.join(METRICS, f"nested_regression_errors_{regime}.npz"))
    trip = np.load(os.path.join(METRICS, f"nested_triplet_errors_{regime}.npz"))
    for name, arr in [("mos", mos), ("trad", trad), ("regr", regr), ("trip", trip)]:
        assert (arr["timestamps"] == mos["timestamps"]).all(), f"{name} timestamps differ"
        assert (arr["obs"] == mos["obs"]).all(), f"{name} obs differ"
    obs = mos["obs"]
    raw = mos["raw_fcst"]

    fcst = {
        "Raw NWP": raw,
        "MOS": mos["mos"],
        "Traditional AnEn": trad["members"].mean(axis=1),
        "Deep-AnEn-Regression": regr["members"].mean(axis=1),
        "Deep Analog (Hu et al., 2023)": trip["members"].mean(axis=1),
    }
    members = {
        "Traditional AnEn": trad["members"],
        "Deep-AnEn-Regression": regr["members"],
        "Deep Analog (Hu et al., 2023)": trip["members"],
    }
    crps = {
        "Raw NWP": mae(raw, obs),   # deterministic: CRPS = MAE
        "MOS": mae(mos["mos"], obs),
        "Traditional AnEn": crps_ens(obs, trad["members"]),
        "Deep-AnEn-Regression": crps_ens(obs, regr["members"]),
        "Deep Analog (Hu et al., 2023)": crps_ens(obs, trip["members"]),
    }

    scores = {}
    for m in METHODS:
        scores[m] = {
            "rmse_kmh": round(rmse(fcst[m], obs), 2),
            "mae_kmh": round(mae(fcst[m], obs), 2),
            "crps_kmh": round(crps[m], 2),
        }
    print(f"[{regime}] " + " | ".join(
        f"{m}: {scores[m]['rmse_kmh']:.2f}/{scores[m]['crps_kmh']:.2f}"
        for m in METHODS))

    calib = {m: calibration(obs, members[m]) for m in members}
    for m, c in calib.items():
        print(f"[{regime}] {m}: PI90 coverage {c['pi90_coverage']:.3f} | "
              f"spread-skill {c['spread_skill_ratio']:.3f} | "
              f"rank counts {c['rank_histogram_counts']}")

    # --- Diebold-Mariano tests vs Traditional AnEn and vs Raw NWP ------
    dm = {}
    for ref_name, ref_err in [("Traditional AnEn", fcst["Traditional AnEn"] - obs),
                              ("Raw NWP", raw - obs)]:
        dm[f"vs_{ref_name}"] = {}
        for m in METHODS:
            if m == ref_name:
                continue
            res = diebold_mariano(fcst[m] - obs, ref_err)
            dm[f"vs_{ref_name}"][m] = {
                "dm_statistic": round(res["dm_statistic"], 3),
                "p_value": res["p_value"],
                "mean_abs_error_diff_kmh": round(res["mean_diff"], 3),
                "n": res["n"], "lag_h": res["lag_h"],
            }
            print(f"[{regime}] DM {m} vs {ref_name}: "
                  f"DM={res['dm_statistic']:+.3f} p={res['p_value']:.4g} "
                  f"mean|e| diff={res['mean_diff']:+.3f}")

    # --- Paired block bootstrap (episode blocks) -------------------------
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    # Per-event squared errors and CRPS contributions, per method.
    sq_err = {m: (fcst[m] - obs) ** 2 for m in METHODS}
    abs_err = {m: np.abs(fcst[m] - obs) for m in METHODS}
    crps_contrib = {
        "Raw NWP": np.abs(raw - obs),
        "MOS": np.abs(mos["mos"] - obs),
    }
    for m in members:
        # Per-event ensemble CRPS (properscoring, one row at a time).
        crps_contrib[m] = np.array([
            properscoring.crps_ensemble(obs[i:i + 1], members[m][i:i + 1])[0]
            for i in range(len(obs))])
    boot = {}
    for ref_name in ["Traditional AnEn", "Raw NWP"]:
        boot[f"vs_{ref_name}"] = {}
        for m in METHODS:
            if m == ref_name:
                continue
            def d_rmse(idx, m=m, ref_name=ref_name):
                return float(np.sqrt(sq_err[m][idx].mean())
                             - np.sqrt(sq_err[ref_name][idx].mean()))
            def d_crps(idx, m=m, ref_name=ref_name):
                return float(crps_contrib[m][idx].mean()
                             - crps_contrib[ref_name][idx].mean())
            boot[f"vs_{ref_name}"][m] = {
                "rmse_diff_kmh": block_bootstrap_ci(d_rmse, ep_ids, n_episodes, rng),
                "crps_diff_kmh": block_bootstrap_ci(d_crps, ep_ids, n_episodes, rng),
            }
            b = boot[f"vs_{ref_name}"][m]
            print(f"[{regime}] Bootstrap {m} vs {ref_name}: "
                  f"dRMSE {b['rmse_diff_kmh']['mean']:+.3f} "
                  f"[{b['rmse_diff_kmh']['ci_lower']:+.3f}, {b['rmse_diff_kmh']['ci_upper']:+.3f}] | "
                  f"dCRPS {b['crps_diff_kmh']['mean']:+.3f} "
                  f"[{b['crps_diff_kmh']['ci_lower']:+.3f}, {b['crps_diff_kmh']['ci_upper']:+.3f}]")

    return {
        "regime": regime,
        "n_test_high_gust_events": int(len(obs)),
        "n_episodes": n_episodes,
        "scores": scores,
        "calibration": calib,
        "diebold_mariano": dm,
        "block_bootstrap": boot,
    }


def main():
    all_results = {
        "method": "Nested-archive comparison (Phase B)",
        "test_period": "2025-11-01 00:00 -> 2026-11-01 23:00 UTC (fixed for all archives)",
        "validation_period": "2025-10-01 -> 2025-10-31 UTC",
        "n_test_high_gust_events": 1286,
        "n_episodes_declustering_gap_24h": 81,
        "bootstrap": {"blocks": "independent episodes (gap >= 24 h between events)",
                      "B": BOOTSTRAP_B, "seed": BOOTSTRAP_SEED,
                      "ci": "percentile 95%"},
        "dm_variant": "d = |e_method| - |e_reference| on per-event absolute errors; "
                      "Newey-West HAC, Bartlett kernel, lag h = floor(T^(1/3)); "
                      "two-sided normal p; negative mean diff means the method beats the reference",
        "regimes": {},
    }
    table_rows = []
    for regime in NESTED_REGIMES:
        res = analyse_regime(regime)
        all_results["regimes"][regime] = res
        for m in METHODS:
            row = {"regime": regime, "method": m,
                   "rmse_kmh": res["scores"][m]["rmse_kmh"],
                   "mae_kmh": res["scores"][m]["mae_kmh"],
                   "crps_kmh": res["scores"][m]["crps_kmh"]}
            # Bootstrap CIs vs Traditional AnEn (the primary reference).
            if m != "Traditional AnEn":
                b = res["block_bootstrap"]["vs_Traditional AnEn"][m]
                row["d_rmse_vs_trad_mean"] = b["rmse_diff_kmh"]["mean"]
                row["d_rmse_vs_trad_ci_lower"] = b["rmse_diff_kmh"]["ci_lower"]
                row["d_rmse_vs_trad_ci_upper"] = b["rmse_diff_kmh"]["ci_upper"]
                row["d_crps_vs_trad_mean"] = b["crps_diff_kmh"]["mean"]
                row["d_crps_vs_trad_ci_lower"] = b["crps_diff_kmh"]["ci_lower"]
                row["d_crps_vs_trad_ci_upper"] = b["crps_diff_kmh"]["ci_upper"]
                row["dm_vs_trad_statistic"] = res["diebold_mariano"]["vs_Traditional AnEn"][m]["dm_statistic"]
                row["dm_vs_trad_p"] = res["diebold_mariano"]["vs_Traditional AnEn"][m]["p_value"]
            table_rows.append(row)

    with open(os.path.join(METRICS, "nested_comparison_metrics.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    boot_only = {r: all_results["regimes"][r]["block_bootstrap"] for r in NESTED_REGIMES}
    with open(os.path.join(METRICS, "nested_bootstrap_cis.json"), "w") as f:
        json.dump(boot_only, f, indent=2)
    table = pd.DataFrame(table_rows)
    table.to_csv(os.path.join(METRICS, "nested_table3.csv"), index=False)
    print("\nSaved metrics/nested_comparison_metrics.json, "
          "metrics/nested_bootstrap_cis.json, metrics/nested_table3.csv")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
