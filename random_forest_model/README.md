# Random Forest temperature forecast

We predict hourly RDU temperature for September 17–30, 2026 with a random forest. We use the same nine inputs as `linear_model/linear_model_1` so the comparison changes the model while holding the available information fixed.

## Run

Use a Python environment with pandas and scikit-learn installed. From the repository root, generate the shared datasets if needed, then run the model:

```sh
python Data/merge_and_split_data.py
python random_forest_model/random_forest_model.py
```

If the three split files already exist in `Data/`, only the second command is needed. Paths are resolved from the script location, so the model can also be run using an absolute path from another working directory.

## Inputs and time split

The model reads these existing pipeline outputs:

| File in `Data/` | Period | Purpose |
|---|---|---|
| `train_dataset_2024_aug2026.csv` | Jan 1, 2024–Aug 31, 2026 | Fit the imputer and forest |
| `val_dataset_sept01_sept16_2026.csv` | Sep 1–16, 2026 | Compute validation errors |
| `inference_dataset_sept17_sept30_2026.csv` | Sep 17–30, 2026 | Generate 336 hourly predictions |

The target is `rdu_tmpf`, in degrees Fahrenheit. Inputs are:

```python
[
    "rdu_tmpf_lag_336h",
    "gfs_tmpf",
    "gfs_dwpf",
    "toa_irradiance",
    "daylength_hours",
    "sin_day",
    "cos_day",
    "sin_hour",
    "cos_hour",
]
```

A forecast input must be available when the forecast is made. For a 14-day window, a 336-hour temperature lag satisfies this constraint: September 17 00:00 uses September 3 00:00, and September 30 23:00 uses September 16 23:00. We mask final-window target values before constructing lags and pass only the nine named features to the model. Current station observations, short lags, gradients, and temperature residuals are excluded from the model inputs.

The script checks that the combined timeline contains every expected hour exactly once before shifting rows. It uses the shared pipeline's naive local-time convention; this is not an elapsed-hour UTC timeline across daylight-saving transitions. The September prediction window has no such transition.

GFS issue-time eligibility must be enforced upstream. The model does not independently reconstruct forecast provenance from the generated split files.

## Fitting and evaluation

We use 300 trees with `max_depth=15`, `min_samples_leaf=2`, `max_features=0.8`, `random_state=42`, and `n_jobs=-1`.

The feature set matches Linear Model 1, but missing-value handling differs: Linear Model 1 drops rows with missing features, while this model imputes them. Training samples therefore differ, so the comparison does not isolate model class alone.

Missing feature values are replaced with medians learned from labeled training rows. Rows with missing targets cannot be used for fitting or scoring. We fit once on the training period, score the validation period, and use that same fitted model for final predictions. We do not refit on validation data or use final test labels to select parameters.

The local run with pandas 2.3.1 and scikit-learn 1.8.0 produced:

| Validation model | MAE (°F) | RMSE (°F) |
|---|---:|---:|
| Random Forest | 2.00 | 2.99 |
| Raw GFS | 2.33 | 3.29 |

Both scores use 384 validation hours in this dataset. The training split contains 23,376 hours; two missing target values leave 23,374 fitting rows. These are validation results, not final September 17–30 test scores.

Validation does not reproduce the final forecast horizon. Historical GFS anchor records have 3–24 hour lead times, whereas the final forecast uses one cycle with anchor lead times of 10–345 hours. In addition, the final 48 validation hours use lagged temperatures from the first two validation days. This evaluates rolling information availability rather than a single 16-day forecast issued on September 1.

## Outputs

The script writes two files to `Data/` and prints validation scores:

- `random_forest_predictions.csv`: `valid_edt` and `predicted_tmpf` for the 336 final hours, in °F.
- `random_forest_feature_importance.csv`: impurity-based importance for the nine inputs. Importance measures how the forest uses features; it does not establish causal effects.

Rerunning the script replaces these two generated files. The source CSVs and shared split files are read-only inputs to this script.
