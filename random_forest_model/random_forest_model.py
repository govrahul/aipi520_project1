"""Random Forest model for the RDU hourly temperature project.

We keep the inputs fixed to measure what changes when we replace linear
regression with a forest of regression trees.

After generating the shared datasets, run from the repository root:
    python random_forest_model/random_forest_model.py
An absolute script path also works from any directory.
"""

from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "Data"

TRAIN_PATH = DATA_DIR / "train_dataset_2024_aug2026.csv"
VAL_PATH = DATA_DIR / "val_dataset_sept01_sept16_2026.csv"
TEST_PATH = DATA_DIR / "inference_dataset_sept17_sept30_2026.csv"
OUTPUT_PATH = DATA_DIR / "random_forest_predictions.csv"
IMPORTANCE_PATH = DATA_DIR / "random_forest_feature_importance.csv"

VAL_START = pd.Timestamp("2026-09-01 00:00:00")
TEST_START = pd.Timestamp("2026-09-17 00:00:00")
TEST_END = pd.Timestamp("2026-09-30 23:00:00")

TARGET = "rdu_tmpf"
LAG_HOURS = 14 * 24
LAG_FEATURE = f"rdu_tmpf_lag_{LAG_HOURS}h"

# Holding the inputs fixed separates the effect of the model from feature changes.
FEATURES = [
    LAG_FEATURE,
    "gfs_tmpf",
    "gfs_dwpf",
    "toa_irradiance",
    "daylength_hours",
    "sin_day",
    "cos_day",
    "sin_hour",
    "cos_hour",
]


def load_datasets() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the shared chronological train, validation, and inference splits."""
    missing = [path for path in [TRAIN_PATH, VAL_PATH, TEST_PATH] if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing generated dataset(s). Run Data/merge_and_split_data.py first: "
            + ", ".join(str(path) for path in missing)
        )

    train = pd.read_csv(TRAIN_PATH)
    val = pd.read_csv(VAL_PATH)
    test = pd.read_csv(TEST_PATH)

    for frame in [train, val, test]:
        frame["valid_edt"] = pd.to_datetime(frame["valid_edt"])

    return train, val, test


def add_14_day_lag(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create a 336-hour lag without using any post-cutoff observations."""
    train = train.copy()
    val = val.copy()
    test = test.copy()

    # At forecast time, temperatures in the prediction window are unknown.
    # Remove these labels even when the input file also contains evaluation data.
    test[TARGET] = float("nan")

    full = pd.concat([train, val, test], ignore_index=True)
    full = full.sort_values("valid_edt").reset_index(drop=True)
    # A row shift represents hours only when each hour occurs exactly once.
    expected_hours = pd.date_range("2024-01-01", TEST_END, freq="h")
    if not pd.DatetimeIndex(full["valid_edt"]).equals(expected_hours):
        raise ValueError("Expected a unique, continuous hourly timeline for the 336h lag.")
    # Availability depends on the timestamp, not which file a row came from.
    full.loc[full["valid_edt"] >= TEST_START, TARGET] = float("nan")
    # The final forecast hour uses Sep 16 23:00, still before our Sep 17 cutoff.
    full[LAG_FEATURE] = pd.to_numeric(full[TARGET], errors="coerce").shift(LAG_HOURS)

    train_out = full[full["valid_edt"] < VAL_START].copy()
    val_out = full[
        (full["valid_edt"] >= VAL_START) & (full["valid_edt"] < TEST_START)
    ].copy()
    test_out = full[
        (full["valid_edt"] >= TEST_START) & (full["valid_edt"] <= TEST_END)
    ].copy()

    if len(test_out) != 336:
        raise ValueError(f"Expected 336 inference hours, found {len(test_out)}.")
    if test_out[LAG_FEATURE].isna().any():
        raise ValueError("14-day lag is missing for one or more inference hours.")

    return train_out, val_out, test_out


def build_model() -> Pipeline:
    """Construct a reproducible nonlinear model with train-only imputation."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "rf",
                RandomForestRegressor(
                    n_estimators=300,
                    max_depth=15,
                    min_samples_leaf=2,
                    max_features=0.8,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def main() -> None:
    train, val, test = load_datasets()
    train, val, test = add_14_day_lag(train, val, test)

    missing_columns = [
        col
        for col in FEATURES + [TARGET]
        if col not in train.columns or col not in val.columns
    ]
    if missing_columns:
        raise ValueError(f"Required columns are missing: {sorted(set(missing_columns))}")

    # A missing target gives us no observed outcome to learn from or score.
    # Feature medians are learned during fit, using only the training rows.
    train_scored = train.dropna(subset=[TARGET]).copy()
    val_scored = val.dropna(subset=[TARGET]).copy()

    model = build_model()
    model.fit(train_scored[FEATURES], train_scored[TARGET])

    # Score later observations to measure performance beyond the training period.
    val_pred = model.predict(val_scored[FEATURES])
    val_mae = mean_absolute_error(val_scored[TARGET], val_pred)
    val_rmse = mean_squared_error(val_scored[TARGET], val_pred) ** 0.5

    print("Random Forest validation")
    print(f"  Hours: {len(val_scored):,}")
    print(f"  MAE : {val_mae:.2f} F")
    print(f"  RMSE: {val_rmse:.2f} F")

    # Compare with GFS directly to measure the value added by the fitted model.
    gfs_val = val_scored.dropna(subset=["gfs_tmpf"])
    gfs_mae = mean_absolute_error(gfs_val[TARGET], gfs_val["gfs_tmpf"])
    gfs_rmse = mean_squared_error(gfs_val[TARGET], gfs_val["gfs_tmpf"]) ** 0.5
    print("\nRaw GFS validation baseline")
    print(f"  MAE : {gfs_mae:.2f} F")
    print(f"  RMSE: {gfs_rmse:.2f} F")

    rf = model.named_steps["rf"]
    importance = pd.DataFrame(
        {"feature": FEATURES, "importance": rf.feature_importances_}
    ).sort_values("importance", ascending=False)
    importance.to_csv(IMPORTANCE_PATH, index=False)

    print("\nFeature importance")
    print(importance.to_string(index=False, formatters={"importance": "{:.4f}".format}))

    # Keep the fitted model fixed for the final window. The input pipeline must
    # supply GFS forecasts issued before the cutoff; the lag uses earlier observations.
    test["predicted_tmpf"] = model.predict(test[FEATURES])
    test[["valid_edt", "predicted_tmpf"]].to_csv(OUTPUT_PATH, index=False)

    print(f"\nSaved {len(test):,} predictions to {OUTPUT_PATH}")
    print(f"Saved feature importance to {IMPORTANCE_PATH}")


if __name__ == "__main__":
    main()
