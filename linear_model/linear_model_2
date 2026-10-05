# Linear regression with short lags + recursive forecasting
# Predict hourly temperature at RDU for Sep 17 - Sep 30
# Run from the project root: python linear_regression_recursive.py
#
# Idea: after Sep 17 we have no real temperatures, so the lag features are empty.
# We predict one hour at a time and use our own predictions as the lags
# for the following hours.

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 1. Load the three datasets
train = pd.read_csv("Data/train_dataset_2024_aug2026.csv")
val = pd.read_csv("Data/val_dataset_sept01_sept16_2026.csv")
test = pd.read_csv("Data/inference_dataset_sept17_sept30_2026.csv")

# 2. Choose features, based on the EDA
#    RDU lags with the highest correlation in the EDA:
#    1h (0.99), 2h (0.97), 3h (0.94), 24h (0.89), 6h (0.84), 48h (0.83)
lags = [1, 2, 3, 6, 24, 48]
lag_features = []
for k in lags:
    lag_features.append("rdu_tmpf_lag_" + str(k) + "h")

#    Not used: gfs_sknt, gfs_cloud_cover_pct, gfs_estimated_ghi (100% missing)
#    Not used: lags of other stations (we cannot predict those stations)
other_features = [
    "gfs_tmpf",             # GFS forecast temperature (EDA correlation 0.98)
    "gfs_dwpf",             # GFS forecast dew point   (EDA correlation 0.85)
    "toa_irradiance",       # solar radiation at top of atmosphere (-0.79)
    "daylength_hours",      # hours of daylight        (0.74)
    "sin_day", "cos_day",   # day of year  (seasonal pattern in the EDA)
    "sin_hour", "cos_hour", # hour of day  (daily pattern in the EDA)
]
features = lag_features + other_features
target = "rdu_tmpf"  # what we predict: observed RDU temperature

# 3. Train the model on rows with no missing values (real lags are used here)
train_clean = train.dropna(subset=features + [target])
model = LinearRegression()
model.fit(train_clean[features], train_clean[target])


# 4. Recursive forecast: predict hour by hour, feeding predictions back as lags
def recursive_predict(past_temps, df):
    history = list(past_temps)                  # temperatures known so far
    df = df.copy()
    df[other_features] = df[other_features].ffill().bfill()  # fill small gaps
    predictions = []
    for i in range(len(df)):
        row = df.iloc[[i]][features].copy()     # one row, as a DataFrame
        for k in lags:
            row["rdu_tmpf_lag_" + str(k) + "h"] = history[-k]  # k hours ago
        pred = model.predict(row)[0]
        predictions.append(pred)
        history.append(pred)                    # the prediction becomes "history"
    return predictions


# 5. Evaluate on validation (Sep 1 - Sep 16) in two ways
val_clean = val.dropna(subset=features + [target])

# 5a. One-step: uses REAL lags (easy, but not possible for Sep 17 - Sep 30)
one_step_pred = model.predict(val_clean[features])
print("Validation MAE, one-step with real lags:",
      round(mean_absolute_error(val_clean[target], one_step_pred), 2), "F")

# 5b. Recursive: starts from the end of training, uses only its own predictions
#     (this is the honest score, same situation as Sep 17 - Sep 30)
train_temps = train[target].ffill().tail(48)
val["recursive_pred"] = recursive_predict(train_temps, val)
val_scored = val.dropna(subset=[target])
mae = mean_absolute_error(val_scored[target], val_scored["recursive_pred"])
rmse = mean_squared_error(val_scored[target], val_scored["recursive_pred"]) ** 0.5
print("Validation MAE, recursive :", round(mae, 2), "F")
print("Validation RMSE, recursive:", round(rmse, 2), "F")

# 6. Look at the coefficient of each feature
print("\nFeature coefficients:")
for name, coef in zip(features, model.coef_):
    print(name, round(coef, 4))

# 7. Predict Sep 17 - Sep 30, starting from the last 48 real hours (Sep 15 - Sep 16)
val_temps = val[target].ffill().tail(48)
test["predicted_tmpf"] = recursive_predict(val_temps, test)
test[["valid_edt", "predicted_tmpf"]].to_csv(
    "Data/linear_regression_recursive_predictions.csv", index=False
)
print("\nPredictions saved:", len(test), "rows")