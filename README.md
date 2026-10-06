# RDU Hourly Temperature Forecasting

This project builds an hourly temperature dataset for Raleigh-Durham International Airport (RDU), investigates its quality and predictive signals, and trains a Random Forest model to predict RDU air temperature for September 17–30, 2026.

The target is `rdu_tmpf`, measured in degrees Fahrenheit. The final inference window contains 336 hourly predictions. The model is evaluated on a preceding, chronological validation window; the final forecast window is not scored in the current pipeline.

## Problem statement

Can historical station observations, numerical weather prediction (NWP), and deterministic solar/calendar features improve on the raw GFS temperature forecast when estimating hourly RDU temperature?

The repository addresses this as a time-aware regression problem:

- **Training:** January 1, 2024–August 31, 2026.
- **Validation:** September 1–16, 2026.
- **Inference:** September 17–30, 2026.
- **Target:** observed hourly RDU temperature (`rdu_tmpf`, °F).
- **Forecast horizon:** a 14-day temperature lag plus GFS forecast values available from a cycle initialized before the September 17 cutoff.

The split is chronological rather than random so validation follows training in time. Inference-time station observations and features derived from them are masked to prevent using unknown future measurements.

## Initial findings

The exploratory data-quality checks in [`EDA/rdu_temperature_eda.ipynb`](EDA/rdu_temperature_eda.ipynb) cover the training and validation splits; they intentionally do not inspect the inference targets.

- The training split has **23,376 hourly rows and 85 columns**; validation has **384 hourly rows and the same 85 columns**.
- Both splits have chronological, continuous hourly timestamps, with **no duplicate timestamps or exact duplicate rows**.
- The RDU target is present for all 384 validation hours and for 23,374 training hours (two training labels are missing). Observed training temperatures range from **12°F to 104°F**, with a median of **66°F**.
- Temperature is strongly persistent: the one-hour RDU lag has a training correlation of about **0.99** with the target. GFS temperature is also strongly associated with the target (about **0.98** training correlation).
- On the validation window, simple rolling persistence baselines score **1.89°F MAE / 2.54°F RMSE** for the one-hour lag, **3.67°F / 4.98°F** for the 24-hour lag, and **5.64°F / 6.91°F** for the 48-hour lag.
- Several candidate source fields are completely missing in the current splits: `gfs_sknt`, `gfs_cloud_cover_pct`, and `gfs_estimated_ghi`. Treat them as unavailable until their source coverage is corrected; do not infer useful signal from all-null columns.

These EDA results are descriptive and are not a substitute for the fitted model's holdout metrics. In particular, the one-hour persistence baseline has different input availability from the 14-day-lag Random Forest.

## Data sources and generation

The prepared datasets combine three sources:

1. **Surface observations:** hourly aggregates of ASOS observations from the Iowa Environmental Mesonet (IEM) for RDU and four nearby stations: IGX, RWI, GSO, and FAY. The downloader requests temperature, dew point, humidity, wind, precipitation, and pressure fields. Sub-hourly reports are averaged to hourly values, with gaps forward-filled for at most two hours. Station observations at or after **September 17, 2026 00:00 Eastern time** are excluded.
2. **NWP forecasts:** GFS 0.25-degree forecast fields downloaded using Herbie. Historical forecast records are sampled at three-hour intervals and interpolated to hourly values. The inference window uses a GFS cycle initialized at **2026-09-16 18:00 UTC**, before the local prediction cutoff. Temperature and dew point are converted from Kelvin to Fahrenheit, and wind speed from meters per second to knots.
3. **Solar and calendar features:** `pvlib` computes solar position, daylight, top-of-atmosphere irradiance, and clear-sky irradiance at the RDU coordinates (35.8776° N, 78.7875° W). Cyclic hour-of-day and day-of-year encodings are also generated.

The merge script creates a continuous hourly `valid_edt` timeline, converts eligible GFS timestamps to Eastern local time, keeps the latest eligible GFS initialization for each valid hour, and interpolates internal GFS gaps. It engineers multi-station temperature lags (1, 2, 3, 6, 12, 24, and 48 hours), regional temperature gradients, dew-point depression, estimated irradiance, and GFS temperature residuals. At inference time, current station observations, observation-derived gradients, and residual targets are masked.

The data pipeline stores Eastern timestamps as timezone-naive local timestamps for joining. This is convenient for the September windows used here, but it is **not an unambiguous elapsed-hour timeline across daylight-saving transitions**.

### Generated files

| File | Description |
|---|---|
| `Data/station_historical_data.csv` | Hourly multi-station ASOS observations from IEM. |
| `Data/gfs_rdu_historical_and_forecast_2024_2026.csv` | Hourly interpolated GFS historical and forecast data at RDU. |
| `Data/rdu_solar_features_2024_2026.csv` | Deterministic hourly solar and calendar features. |
| `Data/train_dataset_2024_aug2026.csv` | Training features and labels (23,376 rows). |
| `Data/val_dataset_sept01_sept16_2026.csv` | Validation features and labels (384 rows). |
| `Data/inference_dataset_sept17_sept30_2026.csv` | Inference features for the 336-hour forecast window. |
| `Data/random_forest_predictions.csv` | Generated predictions (`valid_edt`, `predicted_tmpf`). |
| `Data/random_forest_feature_importance.csv` | Random Forest impurity-based feature importances. |

## Reproduce the work

Run commands from the repository root. Use Python 3 with the dependencies in [`requirements.txt`](requirements.txt).

### 1. Set up the environment

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

### 2. Build the features and splits

If the checked-in/raw source CSVs are already available in `Data/`, regenerate solar features, merge the sources, and write the three time splits:

```bash
python Data/generate_solar_features.py
python Data/merge_and_split_data.py
```

The merge step expects these inputs to exist in `Data/`:

- `station_historical_data.csv`
- `gfs_rdu_historical_and_forecast_2024_2026.csv`
- the solar CSV generated by the first command

To refresh the downloaded source data as well, run the network-backed downloaders before generating solar features and merging:

```bash
python Data/iem_data_load.py
python Data/nwp_predictions_data_load.py
python Data/generate_solar_features.py
python Data/merge_and_split_data.py
```

The download scripts contact external IEM and GFS sources and may take time or return partial data if requests fail. Inspect their completion messages and the generated CSVs before relying on refreshed datasets.

### 3. Train, evaluate, and generate predictions

```bash
python random_forest_model/random_forest_model.py
```

The script prints validation MAE and RMSE, compares those metrics with raw GFS temperature, prints feature importances, and writes the two Random Forest output CSVs listed above. It does not require network access once the split datasets are present.

### 4. Re-run the exploratory analysis

```bash
jupyter lab EDA/rdu_temperature_eda.ipynb
```

The notebook loads the training and validation split CSVs. Generate those files first if they are not already present.

## Model and performance

The implemented model is a scikit-learn `RandomForestRegressor` inside a pipeline that imputes missing feature values with medians learned from the training rows. It uses 300 trees, `max_depth=15`, `min_samples_leaf=2`, `max_features=0.8`, and `random_state=42`. Rows with missing target values are omitted from fitting and scoring.

The nine model inputs are:

- RDU temperature lagged by **336 hours** (`rdu_tmpf_lag_336h`)
- GFS 2 m temperature and dew point (`gfs_tmpf`, `gfs_dwpf`)
- Top-of-atmosphere irradiance and day length (`toa_irradiance`, `daylength_hours`)
- Cyclic day-of-year and hour-of-day encodings (`sin_day`, `cos_day`, `sin_hour`, `cos_hour`)

The 336-hour lag is constructed only from temperatures available before each forecast time. For example, the prediction at September 30 23:00 uses the observed September 16 23:00 temperature. The model checks that the joined timeline contains every expected hour exactly once before constructing the lag.

Recorded validation results are:

| Validation method | Hours scored | MAE (°F) | RMSE (°F) |
|---|---:|---:|---:|
| Random Forest | 384 | **2.00** | **2.99** |
| Linear Model 1 | 384 | 2.04 | 2.96 |
| Raw GFS temperature | 384 | 2.33 | 3.29 |

On the reported validation results, Random Forest has slightly lower MAE than Linear Model 1 (2.00°F vs. 2.04°F), while Linear Model 1 has slightly lower RMSE (2.96°F vs. 2.99°F). Both fitted models outperform raw GFS on MAE and RMSE; relative to raw GFS, Random Forest improves MAE by **0.33°F** and RMSE by **0.30°F**. These are validation—not final inference—scores. The current pipeline does not score the September 17–30 forecast because inference labels are masked/held out.

### How to interpret the comparison

- The Random Forest uses train-fitted median imputation. Linear Model 1 may use different missing-value handling, features, or validation forecasting procedure, so these reported metrics should not be treated as a controlled comparison of model classes without checking that setup.
- The validation GFS fields have historical issue lead times of roughly 3–24 hours, while the final forecast uses a single GFS cycle with lead times of roughly 10–345 hours. The validation metric therefore does not directly establish performance over the full final forecast horizon.
- The final 48 validation hours use temperature lags from earlier validation observations. This is a rolling-availability evaluation, not a single 16-day forecast issued at the start of validation.
- Feature importance is impurity-based and indicates how the fitted forest uses inputs; it is not evidence of causation.
- Forecast quality depends on GFS issue-time eligibility being enforced in the upstream data pipeline. The model reads the prepared files and does not independently reconstruct forecast provenance.

## Repository map

```text
Data/
  iem_data_load.py                   # Fetch and aggregate IEM station observations
  nwp_predictions_data_load.py       # Extract and interpolate GFS forecast fields
  generate_solar_features.py         # Generate pvlib solar/calendar features
  merge_and_split_data.py            # Merge, engineer features, and create time splits
EDA/
  rdu_temperature_eda.ipynb          # Data-quality checks and exploratory analysis
linear_model/
  linear_model_1.py                  # Fit, validate, and generate inference predictions without recursive forecasting
  linear_model_2.py                  # Fit, validate, and generate inference predictions with recursive forecasting
random_forest_model/
  random_forest_model.py             # Fit, validate, and generate inference predictions
  README.md                          # Model-specific implementation notes
requirements.txt                     # Python dependencies
```
