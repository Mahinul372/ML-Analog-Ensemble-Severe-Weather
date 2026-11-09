"""Tables 1 and 2 for the nested-archive revision (Phase B).

Table 1 — data sources, meteorological variables, temporal coverage,
missing-data statistics. Missing-data figures come from the
2026-11-09 no-interpolation probe: the full 2018-01-01 -> 2026-11-01
window re-fetched with a fresh cache and no interpolation returned
77,448 rows with 0 NaN cells in any column, identical to the files in
use (max absolute difference 0.0 in all 8 columns).

Table 2 — forecasting methods, model configurations, hyperparameters.

Outputs: metrics/nested_table1.csv/.json, metrics/nested_table2.csv/.json.

Usage: python nested_tables.py
"""

import json
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
METRICS = os.path.join(HERE, "metrics")
os.makedirs(METRICS, exist_ok=True)

TABLE1_ROWS = [
    {"source": "Open-Meteo Previous-Runs API (previous-runs-api.open-meteo.com/v1/forecast)",
     "role": "NWP forecast (predictors)",
     "variables": "temperature_2m, wind_speed_10m, wind_gusts_10m, surface_pressure",
     "units": "degC, km/h, km/h, hPa",
     "temporal_coverage": "2018-01-01 00:00 -> 2026-11-01 23:00 UTC (hourly)",
     "rows": 77448,
     "missing_values": 0,
     "notes": "No models parameter passed: provider default (auto/best-match). "
              "Empirically identical to NOAA GFS (seamless) at sampled windows in "
              "2018, 2021, 2023 and 2025. Plain variables are the Day-0 offset "
              "(most recent run before the valid hour; GFS updates every 6 h, so "
              "effective lead time 0-6 h). No per-hour model attribution or "
              "run-initialisation field is exposed by the endpoint."},
    {"source": "Open-Meteo Archive API (archive-api.open-meteo.com/v1/archive)",
     "role": "Reference observations (target)",
     "variables": "temperature_2m, wind_speed_10m, wind_gusts_10m, surface_pressure",
     "units": "degC, km/h, km/h, hPa",
     "temporal_coverage": "2018-01-01 00:00 -> 2026-11-01 23:00 UTC (hourly)",
     "rows": 77448,
     "missing_values": 0,
     "notes": "ERA5 reanalysis-based reference data, not independent station "
              "observations. Verification is against reanalysis (see limitations "
              "note in the README/notebook)."},
    {"source": "Nested design — fixed test period (all archives)",
     "role": "Evaluation",
     "variables": "obs_gust >= 35 km/h defines high-gust events",
     "units": "km/h",
     "temporal_coverage": "2025-11-01 00:00 -> 2026-11-01 23:00 UTC (8,784 h)",
     "rows": 8784,
     "missing_values": 0,
     "notes": "1,286 high-gust events; 81 independent episodes (declustering gap "
              ">= 24 h between events)."},
    {"source": "Nested design — fixed validation period (all archives)",
     "role": "Early stopping / model selection only",
     "variables": "all",
     "units": "-",
     "temporal_coverage": "2025-10-01 00:00 -> 2025-10-31 23:00 UTC (744 h)",
     "rows": 744,
     "missing_values": 0,
     "notes": "Never used for fitting or for test evaluation."},
    {"source": "Nested design — training periods (identical cutoff 2025-09-30 23:00)",
     "role": "Training pools",
     "variables": "all",
     "units": "-",
     "temporal_coverage": "1-year: 2024-11-01 -> 2025-09-30 (8,016 rows); "
              "3-year: 2022-11-01 -> 2025-09-30 (25,560); "
              "5-year: 2020-11-01 -> 2025-09-30 (43,080); "
              "8-year: 2018-01-01 -> 2025-09-30 (67,920; the 8-year archive uses "
              "the forecast data available from the source, beginning 1 January 2018)",
     "rows": 144576,
     "missing_values": 0,
     "notes": "Nested pools (each longer archive contains the shorter ones' "
              "training rows). Scalers are fitted on each archive's training "
              "data only."},
]

TABLE2_ROWS = [
    {"method": "Raw NWP",
     "type": "Deterministic baseline",
     "configuration": "Forecast gust (fcst_gust) used directly, no post-processing",
     "hyperparameters": "-",
     "similarity_space": "-"},
    {"method": "MOS (Model Output Statistics)",
     "type": "Deterministic baseline",
     "configuration": "Ordinary least-squares linear regression of obs_gust on the "
                      "four standardized forecast predictors; fitted on each "
                      "archive's training data only",
     "hyperparameters": "OLS, closed form; intercept + 4 coefficients",
     "similarity_space": "-"},
    {"method": "Traditional AnEn",
     "type": "Analog ensemble (Delle Monache et al. 2011, 2013)",
     "configuration": "Euclidean nearest neighbours in standardized forecast space; "
                      "K=15 members, +/-168 h exclusion, training pool only",
     "hyperparameters": "K=15; exclusion 168 h",
     "similarity_space": "4-D (standardized forecasts)"},
    {"method": "Deep-AnEn-Regression",
     "type": "Learned analog ensemble",
     "configuration": "PyTorch encoder Linear 4->32 -> ReLU -> Dropout(0.2) -> "
                      "Linear 32->16 latent + Linear 16->1 training head; "
                      "K=15 nearest training embeddings, +/-168 h exclusion",
     "hyperparameters": "Adam lr 1e-3; MSE vs standardized obs_gust; 30 epochs; "
                        "full-batch; seed 42",
     "similarity_space": "16-D latent"},
    {"method": "Deep Analog (Hu et al., 2023)",
     "type": "Learned analog ensemble (triplet network)",
     "configuration": "3-layer LSTM (20 hidden, dropout 0.01) + linear layer to "
                      "20-D embeddings; TripletMarginLoss; Reverse Analog "
                      "fitness-proportionate sampling; 24 h lookback sequences; "
                      "K=15 retrieval, +/-168 h exclusion, training pool only",
     "hyperparameters": "Adam lr 0.005; margin 0.9 (official default); no L2 "
                        "normalization; ReduceLROnPlateau (factor 0.2, patience 5); "
                        "early stopping patience 15, max 100 epochs; 2,000 anchors "
                        "per epoch -> 30,000 triplets; batch 256; seed 42",
     "similarity_space": "20-D embedding"},
]


def main():
    t1 = pd.DataFrame(TABLE1_ROWS)
    t1.to_csv(os.path.join(METRICS, "nested_table1.csv"), index=False)
    with open(os.path.join(METRICS, "nested_table1.json"), "w") as f:
        json.dump(TABLE1_ROWS, f, indent=2)
    t2 = pd.DataFrame(TABLE2_ROWS)
    t2.to_csv(os.path.join(METRICS, "nested_table2.csv"), index=False)
    with open(os.path.join(METRICS, "nested_table2.json"), "w") as f:
        json.dump(TABLE2_ROWS, f, indent=2)
    print("Saved metrics/nested_table1.csv/.json, metrics/nested_table2.csv/.json")
    print(t1.to_string(index=False))
    print(t2.to_string(index=False))


if __name__ == "__main__":
    main()
