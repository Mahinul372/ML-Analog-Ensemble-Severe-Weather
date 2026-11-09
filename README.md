# ML Analog Ensemble for Severe-Weather Gust Forecasting

**Question.** Can analog-ensemble methods beat raw numerical weather prediction (NWP) for forecasting severe wind gusts, and does a learned similarity space help when the archive gets longer? **Answer.** Yes to the first — every analog method beats raw NWP by a wide margin in both regimes — and no, not as configured, to the second: the simplest 4-D analog search stays best, and the learned spaces do not overtake it even on 8 years 10 months of data.

## Dataset

Hourly data for Atlanta, GA (33.64, −84.42), built from two Open-Meteo sources: **forecasts** from the Previous-Runs API (historical NWP forecasts) and **observations** from the Archive API (ERA5). Four forecast predictors — temperature, wind speed, gust, surface pressure — and the observed gust (`obs_gust`) as the target. The archive starts **2018-01-01**: the Previous-Runs API returns no valid forecast data earlier (verified by probe, 2026-11-09), so 2018-01-01 is the earliest usable date.

The long archive is stored as **yearly files** because the combined 7.15 MB file exceeded the GitHub snapshot upload size limit; concatenating them reproduces it exactly. The 1-year file is `data/atlanta_weather_archive_1yr.csv`.

| File | Date range | Rows |
|---|---|---|
| `data/atlanta_weather_archive_1yr.csv` | 2025-11-02 – 2026-11-01 | 8,760 |
| `data/atlanta_weather_archive_2018.csv` | 2018-01-01 – 2018-12-31 | 8,760 |
| `data/atlanta_weather_archive_2019.csv` | 2019-01-01 – 2019-12-31 | 8,760 |
| `data/atlanta_weather_archive_2020.csv` | 2020-01-01 – 2020-12-31 | 8,784 |
| `data/atlanta_weather_archive_2021.csv` | 2021-01-01 – 2021-12-31 | 8,760 |
| `data/atlanta_weather_archive_2022.csv` | 2022-01-01 – 2022-12-31 | 8,760 |
| `data/atlanta_weather_archive_2023.csv` | 2023-01-01 – 2023-12-31 | 8,760 |
| `data/atlanta_weather_archive_2024.csv` | 2024-01-01 – 2024-12-31 | 8,784 |
| `data/atlanta_weather_archive_2025.csv` | 2025-01-01 – 2025-12-31 | 8,760 |
| `data/atlanta_weather_archive_2026.csv` | 2026-01-01 – 2026-11-01 | 7,320 |

## Regimes

- **1-year:** 2025-11-02 to 2026-11-01 (8,760 rows). Train ≤ 2026-08-15 (6,888 rows), validation to 2026-08-31 (384), test ≥ 2026-09-01 (1,488). **130 test storm events** (observed gust ≥ 35 km/h).
- **8 years 10 months:** 2018-01-01 to 2026-11-01 (77,448 rows). Train 2018-01-01 to 2024-11-01 (59,928 rows), validation 1 year (8,760), test 1 year (8,760). **1,286 test storm events.**

`_8yr10mo` = 8 years 10 months. All splits are strictly chronological; analogs are drawn from the training pool only, with a ±168 h (±7-day) exclusion around each target.

## Methods

- **Raw NWP** — the forecast gust as issued, no correction.
- **Traditional AnEn** — analog search in the 4-D standardized predictor space (K = 15).
- **Deep-AnEn-Regression** — a 4→32→16 MLP embeds forecasts into a 16-D latent space; analogs are retrieved there (K = 15).
- **Deep Analog (Hu et al., 2023)** — a 3-layer LSTM embeds 24-hour forecast sequences into a 20-D space learned with a triplet network (K = 15).

## Results

RMSE and CRPS (km/h) on the test storm events. Diebold–Mariano p-values are against Traditional AnEn (p < 0.05 = significant). The long regime is the 8-year 10-month archive (2018–2026); table headers abbreviate it as "8yr 10mo".

| Method | 1-yr RMSE | 1-yr CRPS | 1-yr DM p | 8yr 10mo RMSE | 8yr 10mo CRPS | 8yr 10mo DM p |
|---|---:|---:|---:|---:|---:|---:|
| Raw NWP | 13.23 | 11.42 | 5.6×10⁻¹⁰ | 12.79 | 10.87 | < 0.001 |
| Traditional AnEn | 7.43 | 4.15 | — | 7.63 | 4.36 | — |
| Deep-AnEn-Regression | 7.86 | 4.45 | 0.035 | 7.67 | 4.41 | 0.227 |
| Deep Analog (Hu et al., 2023) | 8.35 | 5.15 | 0.029 | 9.39 | 5.46 | 7.0×10⁻⁶ |

**Two findings.** First, Deep-AnEn-Regression **caught up** on the long archive: its gap to Traditional AnEn shrank from 0.43 to 0.04 km/h RMSE and the difference is no longer significant (p = 0.227) — the two are statistically tied. Second, the ordering **persisted**: skill falls as the similarity space grows, 4-D > 16-D > 20-D, and the triplet model's gap to Traditional widened (0.92 → 1.76 km/h RMSE, p = 7.0×10⁻⁶). The earlier prediction that a longer archive would narrow or reverse the ordering is falsified for the triplet model as configured — its validation loss barely improved on the long run, so the configuration, not data volume, looks like the binding constraint. This matches Platzer et al. (2024): the optimal analog distance depends on data volume, and a reduced feature space preserves analog performance on small-to-medium archives. The same pattern holds when the long-trained models are scored on the exact 130 1-year events (Traditional 7.02, Regression 7.10, Deep Analog 8.31 km/h RMSE).

Verification: the comparison notebook audits both regimes — 5,850 (1-year) and 57,870 (long) analog selections, all six checks passed (identical training pool, ±168 h exclusion, members equal to training observations, identical scalers, saved metrics reproduce exactly).

![Four-way method comparison, 1-year regime](plots/4way_method_comparison_timeseries.svg)

![Dual-regime RMSE / CRPS benchmark](plots/dual_regime_rmse_crps_barchart.svg)

## Training vs validation loss

Overfitting would show as a training loss that keeps falling while the validation loss turns up and diverges. None of the four curves shows that: in every case the validation loss falls alongside the training loss and the final train–validation gap is small (under ~0.1). The regression curves end with validation below training loss — the standard dropout artefact, not a sign of anything wrong. Traditional AnEn has no loss curve, by design: it is non-parametric, with no learned parameters and nothing to overfit in the neural sense.

**Deep-AnEn-Regression, 1-year (30 epochs):**

![Deep-AnEn-Regression train/val loss, 1-year](plots/train_val_loss_regression_1yr.svg)

**Deep-AnEn-Regression, 8 years 10 months (30 epochs):**

![Deep-AnEn-Regression train/val loss, 8 years 10 months](plots/train_val_loss_regression_8yr10mo.svg)

**Deep Analog (Hu et al., 2023), 1-year (100 epochs):**

![Deep Analog train/val loss, 1-year](plots/train_val_loss_triplet_1yr.svg)

**Deep Analog (Hu et al., 2023), 8 years 10 months (100 epochs max, early stop at 45):**

![Deep Analog train/val loss, 8 years 10 months](plots/train_val_loss_triplet_8yr10mo.svg)

## How to reproduce

```bash
python fetch_data.py                  # 1-year archive (2025-11-02 – 2026-11-01)
python fetch_data.py --long-archive   # 8 years 10 months archive, yearly files
python build_comparison_notebook.py   # rebuild notebooks/comparison_analysis.ipynb
python -m nbconvert --to notebook --execute --inplace notebooks/comparison_analysis.ipynb
```

The notebook re-runs all four methods from the saved checkpoints and re-verifies both audits; no retraining is needed. PNG figures and `.pth` weights are committed in local git only — the GitHub snapshot connector is text-only — so the repo carries SVG figures and JSON state dicts instead; running the notebook regenerates the PNGs and `.pth` files locally.

## Project structure

```
├── data/                               # yearly CSVs (see Dataset table)
├── notebooks/
│   ├── deep_anen_severe_weather.ipynb  # Phase 2: train Deep-AnEn-Regression
│   └── comparison_analysis.ipynb       # 4-way comparison, audits, DM tests
├── metrics/                            # saved metrics (JSON; _8yr10mo = long regime)
├── models/                             # checkpoints (.pth local only; .json on GitHub)
├── plots/                              # figures (SVG on GitHub; PNG local only)
├── data_prep.py                        # shared protocol: splits, scaler, events
├── fetch_data.py                       # Open-Meteo data pipeline
├── traditional_anen.py                 # Traditional AnEn engine (K=15)
├── phase4_triplet.py                   # Deep Analog (Hu et al., 2023), LSTM + triplets
├── train_regression_long.py            # Deep-AnEn-Regression retrain (long regime)
├── dm_tests.py                         # Diebold–Mariano tests, both regimes
├── plot_loss_curves.py                 # training-vs-validation loss figures
├── build_comparison_notebook.py        # builds the comparison notebook
└── README.md
```

## References

- Delle Monache, L., et al. (2011). *Probabilistic Weather Prediction with an Analog Ensemble.* Monthly Weather Review, 139(11).
- Delle Monache, L., et al. (2013). *An Analog Ensemble for NWP Post-Processing.* Monthly Weather Review, 141(11).
- Hu, W., Cervone, G., Young, G., et al. (2023). *Machine Learning Weather Analogs for Near-Surface Variables.* Boundary-Layer Meteorology, 186, 711–735. https://doi.org/10.1007/s10546-022-00779-6 — official implementation: https://github.com/uga-gaim/DeepAnalog
- Platzer, P., et al. (2024). *Distance Learning for Analog Methods.* HAL-04841334.
- Diebold, F. X., & Mariano, R. S. (1995). *Comparing Predictive Accuracy.* Journal of Business & Economic Statistics, 13(3), 253–263.
- Open-Meteo: https://open-meteo.com/ (Previous-Runs & Archive APIs)

## License

MIT — see `LICENSE` when added.
