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
├── fetch_data.py                     # Open-Meteo data pipeline
├── analog_ensemble.py                # AnalogEnsembleEngine (K=10)
├── run_pipeline.py                   # end-to-end runner + verification
├── requirements.txt
└── README.md
```

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
