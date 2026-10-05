# Linear regression with a 14-day lag feature
# Predict hourly temperature at RDU for Sep 17 - Sep 30
# Run from the project root: python linear_regression_lag14d.py

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 1. Load the three datasets
train = pd.read_csv("Data/train_dataset_2024_aug2026.csv")
val = pd.read_csv("Data/val_dataset_sept01_sept16_2026.csv")
test = pd.read_csv("Data/inference_dataset_sept17_sept30_2026.csv")

# 2. Build the 14-day lag (336 hours = 14 days)
#    Sep 30 23:00 minus 14 days = Sep 16 23:00, so every test hour has a value
full = pd.concat([train, val, test])
full["rdu_tmpf_lag_336h"] = full["rdu_tmpf"].shift(336)
train = full.iloc[:len(train)].copy()
val = full.iloc[len(train):len(train) + len(val)].copy()
test = full.iloc[len(train) + len(val):].copy()

# 3. Choose features, based on the EDA
#    Not used: gfs_sknt, gfs_cloud_cover_pct, gfs_estimated_ghi (100% missing)
features = [
    "rdu_tmpf_lag_336h",    # RDU temperature 14 days ago
    "gfs_tmpf",             # GFS forecast temperature (EDA correlation 0.98)
    "gfs_dwpf",             # GFS forecast dew point   (EDA correlation 0.85)
    "toa_irradiance",       # solar radiation at top of atmosphere (-0.79)
    "daylength_hours",      # hours of daylight        (0.74)
    "sin_day", "cos_day",   # day of year  (seasonal pattern in the EDA)
    "sin_hour", "cos_hour", # hour of day  (daily pattern in the EDA)
]
target = "rdu_tmpf"  # what we predict: observed RDU temperature

# 4. Drop rows with missing values (linear regression cannot handle NaN)
train = train.dropna(subset=features + [target])
val = val.dropna(subset=features + [target])

# 5. Train the model
model = LinearRegression()
model.fit(train[features], train[target])

# 6. Evaluate on the validation set (Sep 1 - Sep 16)
val_pred = model.predict(val[features])
mae = mean_absolute_error(val[target], val_pred)
rmse = mean_squared_error(val[target], val_pred) ** 0.5
print("Validation MAE :", round(mae, 2), "F")
print("Validation RMSE:", round(rmse, 2), "F")

# 7. Look at the coefficient of each feature
print("\nFeature coefficients:")
for name, coef in zip(features, model.coef_):
    print(name, round(coef, 4))

# 8. Predict Sep 17 - Sep 30 and save the results
print("\nRows with missing features in test:", test[features].isna().any(axis=1).sum())
test[features] = test[features].ffill().bfill()  # fill any gaps with nearby values
test["predicted_tmpf"] = model.predict(test[features])
test[["valid_edt", "predicted_tmpf"]].to_csv(
    "Data/linear_regression_lag14d_predictions.csv", index=False
)
print("Predictions saved:", len(test), "rows")