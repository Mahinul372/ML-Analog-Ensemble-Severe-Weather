# ML Analog Ensemble for Severe Weather

**Phase 1 — NWP post-processing with the Analog Ensemble (AnEn) method**

A complete, reproducible pipeline that pairs archived numerical weather
prediction (NWP) forecasts with ground observations for Atlanta, GA, and
implements a basic Analog Ensemble search engine that converts a single
deterministic forecast into a calibrated probabilistic ensemble.

---

## Project Overview

Deterministic NWP forecasts are sharp but biased: they systematically
under-predict extreme events such as severe wind gusts. The Analog
Ensemble (AnEn) corrects this by searching a historical archive for past
forecasts that closely resemble the current one and using the *observed*
weather at those analog times as an ensemble. No model retraining is
required — the archive itself is the statistical model.

This repository (Phase 1) delivers:

- **Data pipeline** (`fetch_data.py`) — pulls 365 days of hourly NWP
  forecasts and ERA5 observations for Atlanta, GA from Open-Meteo and
  aligns them into a single clean archive.
- **Analog Ensemble engine** (`analog_ensemble.py`) — standardized
  feature search with a 7-day leakage-exclusion window, returning
  ensemble mean, median, and 10th/90th percentile predictions.
- **Pipeline runner** (`run_pipeline.py`) — end-to-end verification on
  a real severe-weather case (the peak observed wind gust in the
  archive).

## Methodology: Analog Ensemble for NWP Post-Processing

The method follows Delle Monache et al. (2011, 2013). For a target time
*t* with forecast feature vector

$$v(t) = [\text{temp}_{2m},\ \text{wind}_{10m},\ \text{gust}_{10m},\ \text{pressure}_{sfc}]$$

the engine:

1. **Extracts** the forecast vector at *t*.
2. **Excludes** a ±7-day window around *t* from the search archive so
   the target's own weather cannot leak into its analogs.
3. **Standardizes** all forecast features with `StandardScaler`, so
   variables with different units contribute comparably.
4. **Computes** the normalized Euclidean distance between *v(t)* and
   every remaining historical forecast vector.
5. **Selects** the *K* = 10 closest historical timestamps.
6. **Builds the ensemble** from the *observed* values at those analog
   times and reports the ensemble mean, median, 10th and 90th
   percentiles — a calibrated probabilistic forecast for each variable.

**Phase 1 verification (peak-gust case, 2026-01-31 20:00 UTC):**

| | Gust (km/h) |
|---|---|
| Raw NWP forecast | 52.9 |
| Observed | 77.0 |
| AnEn ensemble mean | 54.6 |
| AnEn 10th–90th percentile | 47.4 – 68.8 |

The raw forecast under-predicted the extreme gust by ~24 km/h. The
ensemble mean shows the same direction of bias (expected at K=10 in a
one-year archive), while the 90th percentile (68.8 km/h) moves
substantially toward the observed extreme — the expected signature of a
probabilistic correction. Extending the archive, tuning K, and adding
predictor variables are Phase 2 work.

## Dataset Details

| | |
|---|---|
| **Location** | Atlanta, GA — 33.64° N, 84.42° W |
| **Period** | 2025-11-02 → 2026-11-01 (365 days, hourly, UTC) |
| **Rows** | 8,760 (complete — no missing values after cleaning) |
| **Forecast source** | Open-Meteo Previous-Runs API (archived NWP forecasts) |
| **Observation source** | Open-Meteo Archive API (ERA5 reanalysis) |
| **File** | `data/atlanta_weather_archive.csv` |

Columns:

| Column | Description | Unit |
|---|---|---|
| `timestamp` | Hourly UTC timestamp | — |
| `fcst_temp` / `obs_temp` | Temperature at 2 m | °C |
| `fcst_wind` / `obs_wind` | Wind speed at 10 m | km/h |
| `fcst_gust` / `obs_gust` | Wind gusts at 10 m | km/h |
| `fcst_press` / `obs_press` | Surface pressure | hPa |

Cleaning: short gaps (≤3 h) are linearly interpolated in time with
forward/backward fill; any remaining NaNs are dropped. The fetched
archive required no row drops.

## Quickstart Guide

### 1. Clone and install

```bash
git clone https://github.com/Mahinul372/ML-Analog-Ensemble-Severe-Weather.git
cd ML-Analog-Ensemble-Severe-Weather
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run the full pipeline

```bash
python run_pipeline.py
```

This re-fetches the 365-day archive from Open-Meteo, rebuilds the
engine, and runs the peak-gust test case, printing forecast vs.
observation vs. ensemble metrics.

To reuse the shipped CSV without re-fetching:

```bash
python run_pipeline.py --skip-fetch
```

### 3. Use the engine in your own code

```python
import pandas as pd
from analog_ensemble import AnalogEnsembleEngine

df = pd.read_csv("data/atlanta_weather_archive.csv", parse_dates=["timestamp"])
engine = AnalogEnsembleEngine(df)

result = engine.find_analogs("2026-01-31 20:00:00+00:00", K=10)
print(result["ensemble_mean"])    # mean observed values at analog times
print(result["p10"], result["p90"])
print(result["analogs"])          # the 10 closest historical timestamps
```

### Project structure

```
├── data/
│   └── atlanta_weather_archive.csv   # 8,760 hourly rows, Atlanta GA
├── notebooks/
│   ├── deep_anen_severe_weather.ipynb  # Phase 2 Colab notebook (GPU)
│   └── comparison_analysis.ipynb       # Phase 3 3-way benchmark
├── models/
│   ├── deep_anen_weights.pth         # encoder weights (generated on run)
│   └── deep_anen_weights.json        # text-safe state dict
├── plots/
│   ├── fig1_event_timeseries.png/.svg
│   ├── fig2_metrics_bars.png/.svg
│   ├── 3way_method_comparison_timeseries.png/.svg
│   └── rmse_crps_benchmark_barchart.png/.svg
├── metrics/
│   ├── deep_anen_metrics.json
│   ├── traditional_anen_metrics.json
│   └── comparison_metrics.json
├── data_prep.py                      # shared protocol: split/scaler/events
├── traditional_anen.py               # Traditional AnEn engine (K=15)
├── fetch_data.py                     # Open-Meteo data pipeline
├── analog_ensemble.py                # AnalogEnsembleEngine (K=10)
├── run_pipeline.py                   # end-to-end runner + verification
├── requirements.txt
└── README.md
```

### Run the Phase 2 notebook

Open
[`notebooks/deep_anen_severe_weather.ipynb`](notebooks/deep_anen_severe_weather.ipynb)
in Google Colab, select a GPU runtime (*Runtime → Change runtime type →
T4 GPU*), then *Run all*. It trains the encoder (30 epochs), evaluates on
130 test storm events, and regenerates the figures, weights, and metrics.

## Phase 2 — Deep Analog Ensemble (Deep-AnEn)

**Executive summary.** Phase 1 searched for analogs in standardized *raw*
feature space. Phase 2 learns the similarity metric: a PyTorch encoder maps
the 4-D forecast vector into a 16-D latent space trained (30 epochs, MSE,
Adam) to predict observed wind gusts, and analog search runs in that learned
space. On 130 held-out test storm events (observed gust ≥ 35 km/h,
2026-09-01 → 2026-11-01), Deep-AnEn cuts gust RMSE by **40.6%** and CRPS by
**61.1%** versus raw NWP. Full runnable notebook:
[`notebooks/deep_anen_severe_weather.ipynb`](notebooks/deep_anen_severe_weather.ipynb)
(Google Colab, GPU runtime).

**Model structure — `DeepAnalogEncoder`:**

```
forecast (4) ──▶ Linear(4 → 32) ──▶ ReLU ──▶ Dropout(0.2)
             ──▶ Linear(32 → 16) ──▶ latent embedding z (16-D)
z ──▶ Linear(16 → 1) ──▶ predicted standardized gust (training head only)
```

Analog search: encode the target forecast to `z`, exclude a ±7-day window,
take the K = 15 nearest training-period embeddings (Euclidean), and build
the ensemble from their observed gusts. Splits are strictly chronological
(train ≤ 2026-08-15, validation to 2026-08-31, test from 2026-09-01) and
analogs are drawn from the training archive only.

**Scores — 130 test storm events (obs gust ≥ 35 km/h):**

| Forecast | RMSE (km/h) | CRPS (km/h) |
|---|---|---|
| Raw NWP | 13.23 | 11.42 |
| **Deep-AnEn (K=15)** | **7.86** | **4.45** |
| **Skill improvement** | **+40.6%** | **+61.1%** |

![Severe wind event, 2026-10-10](plots/fig1_event_timeseries.svg)
*Figure 1 — Most severe test event (peak observed gust 71.6 km/h,
2026-10-10 09:00 UTC, ±48 h): raw NWP vs. observed gusts vs. Deep-AnEn
mean with 90% envelope (p5–p95).*

![RMSE and CRPS comparison](plots/fig2_metrics_bars.svg)
*Figure 2 — RMSE and CRPS: raw NWP vs. Deep-AnEn over the 130 test storm
events.*

**Honest limitation.** Deep-AnEn improves the *average* storm markedly, but
the ensemble mean still under-predicts the single most extreme peak (52 vs.
71.6 km/h observed in Figure 1); the 90% envelope reaches ~65 km/h. Extreme
tails remain the hard case — Phase 3 work (longer archive, K tuning,
additional predictors).

**Artifacts:** weights in `models/deep_anen_weights.pth` (generated on run)
with a text-safe JSON state dict at `models/deep_anen_weights.json`;
metrics in `metrics/deep_anen_metrics.json`; figures in `plots/` (PNG 300 dpi
+ SVG). The notebook regenerates all of them end-to-end.

## Phase 3 — Traditional AnEn Benchmark & 3-Way Comparison

**Executive summary.** Phase 2 beat raw NWP by a wide margin — but is the
*learned* latent space actually better than the *classical* Analog Ensemble?
On the exact same 130 untouched test storm events, **Traditional AnEn
narrowly beats Deep-AnEn on both scores**. Both analog methods crush raw
NWP; the learned space adds no advantage on one year of data with four
predictors. That is the finding, and the benchmark exists to make it
checkable: one shared protocol, one timestamp list, recomputed scores.

**Shared protocol (`data_prep.py`).** The dataset, chronological
train/validation/test split, feature scaler, storm-event subset, and
episode rule are defined once in `data_prep.py` and imported by every
experiment — never recreated per method. The scaler is fit on the
training split only. The evaluation subset is the 130 test-window
timestamps with observed gust ≥ 35 km/h, built once and shared by all
three methods (the comparison notebook asserts all three evaluated
identical timestamps). Deep-AnEn scores are recomputed from the saved
weights and checked against the Phase 2 metric file.

**Complete 3-way model evaluation table — 130 test storm events:**

| Method | RMSE | CRPS |
|---|---|---|
| Raw NWP | 13.23 | 11.42 |
| Traditional AnEn | 7.43 | 4.15 |
| Deep-AnEn | 7.86 | 4.45 |

Skill vs. raw NWP: Traditional AnEn +43.9% RMSE / +63.7% CRPS;
Deep-AnEn +40.6% RMSE / +61.1% CRPS.

![3-way severe-episode comparison](plots/3way_method_comparison_timeseries.svg)
*Figure 1 — Severe episode selected by a predefined rule: the contiguous
run of test-window hours with observed gust ≥ 35 km/h that has the
highest observed peak (2026-10-10 01:00–13:00 UTC, 13 h, peak 71.6 km/h),
plotted with ±24 h padding. Ground truth vs. raw NWP vs. Traditional
AnEn mean with 90% prediction interval (p5–p95) vs. Deep-AnEn mean with
90% prediction interval.*

![RMSE and CRPS benchmark](plots/rmse_crps_benchmark_barchart.svg)
*Figure 2 — RMSE and CRPS across all three methods on the identical
130-event test set.*

**Scoring & terminology notes.** CRPS for both AnEn variants is the
empirical ensemble CRPS over the 15 members; for deterministic Raw NWP
it is the mean absolute error, the exact single-member equivalent of
ensemble CRPS. The 90% bands are **prediction intervals** (5th–95th
percentile of the 15 members), not confidence intervals — and with 15
members those percentile estimates are coarse (each member spans ~6.7%
of the probability mass; p5/p95 sit near the 1st and 15th order
statistics).

**Artifacts:** `traditional_anen.py` (classical 4-D Euclidean engine),
`notebooks/comparison_analysis.ipynb` (standalone Colab benchmark),
`metrics/comparison_metrics.json` and
`metrics/traditional_anen_metrics.json`, figures in `plots/` (PNG 300 dpi
+ SVG). Full notebook:
[`notebooks/comparison_analysis.ipynb`](notebooks/comparison_analysis.ipynb).

**Verification audit (§2b of the notebook).** The shared protocol makes
fairness true by construction, but the notebook also checks it explicitly
with asserts over all 130 events × 15 members × 2 methods (3,900 analog
selections): (1) both methods draw from the identical 6,888-row training
pool; (2) the closest selected analog of any event is 709 h from its
target (> 168 h exclusion); (3) every analog is a training timestamp
with its member equal to the training observation at that time — no test
data in analog selection; (4) both scalers (and the saved Phase 2
checkpoint scaler) are identical; (5) Traditional AnEn CRPS recomputed
4.15 vs. saved 4.15 km/h; (6) 130 unique event timestamps, all arrays
finite and (130, 15). All six checks pass.

## Roadmap

- **Phase 2:** CRPS / rank-histogram verification, K and predictor
  sensitivity, multi-year archive, additional locations.
- **Phase 3:** Operational mode — daily AnEn runs on live forecasts.

## References

- Delle Monache, L., et al. (2011). *Probabilistic Weather Prediction
  with an Analog Ensemble.* Monthly Weather Review, 139(11).
- Delle Monache, L., et al. (2013). *An Analog Ensemble for
  NWP Post-Processing.* Monthly Weather Review, 141(11).
- Open-Meteo: https://open-meteo.com/ (Previous-Runs & Archive APIs)

## License

MIT — see `LICENSE` when added.
