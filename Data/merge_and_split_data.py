import os
import numpy as np
import pandas as pd

# Input File Paths
GFS_PATH = "Data/gfs_rdu_historical_and_forecast_2024_2026.csv"
STATION_PATH = "Data/station_historical_data.csv"
SOLAR_PATH = "Data/rdu_solar_features_2024_2026.csv"

# Output File Paths
TRAIN_OUTPUT_PATH = "Data/train_dataset_2024_aug2026.csv"
VAL_OUTPUT_PATH = "Data/val_dataset_sept01_sept16_2026.csv"
INFERENCE_OUTPUT_PATH = "Data/inference_dataset_sept17_sept30_2026.csv"

# Split Boundaries (EDT)
VAL_START_EDT = pd.to_datetime("2026-09-01 00:00:00")
CUTOFF_EDT = pd.to_datetime("2026-09-17 00:00:00")
INFERENCE_END_EDT = pd.to_datetime("2026-09-30 23:00:00")

# Station ID Mapping
STATION_MAP = {
    "RDU": "Raleigh-Durham",
    "IGX": "Chapel Hill",
    "RWI": "Rocky Mount",
    "GSO": "Greensboro",
    "FAY": "Fayetteville",
}


def load_and_pivot_stations(station_csv_path: str) -> pd.DataFrame:
    """Pivots multi-station data from long-format to 1 row per hour (wide-format)."""
    print("Loading and pivoting station observations...")
    df = pd.read_csv(station_csv_path)

    # Standardize time column name
    time_col = [
        c for c in df.columns if "valid" in c or "date" in c or "time" in c
    ][0]
    df["valid_edt"] = pd.to_datetime(df[time_col])
    # Enforce the as-of boundary here as well as in the source downloader.
    df = df[df["valid_edt"] < CUTOFF_EDT].copy()

    # Identify station column ('station', 'station_id', or 'station_code')
    stn_col = [
        c for c in df.columns if "station" in c or "id" in c or "code" in c
    ][0]
    df[stn_col] = df[stn_col].str.upper().str.strip()

    # Numeric observation columns to pivot (tmpf, dwpf, sknt, drct, relh, etc.)
    value_cols = [
        c for c in df.columns if c not in [time_col, "valid_edt", stn_col]
    ]

    # Pivot table: Index = valid_edt, Columns = station_id
    pivoted = df.pivot_table(
        index="valid_edt",
        columns=stn_col,
        values=value_cols,
        aggfunc="first",
    )

    # Flatten multi-level column headers (e.g. ('tmpf', 'RDU') -> 'rdu_tmpf')
    pivoted.columns = [
        f"{stn.lower()}_{var}" for var, stn in pivoted.columns
    ]
    pivoted = pivoted.reset_index()

    return pivoted


def load_and_preprocess_master() -> pd.DataFrame:
    """Loads GFS, Solar, and Pivoted Stations, merging onto a continuous hourly timeline."""
    station_pivoted_df = load_and_pivot_stations(STATION_PATH)

    print("Loading GFS and Solar datasets...")
    gfs_df = pd.read_csv(GFS_PATH)
    solar_df = pd.read_csv(SOLAR_PATH)

    # The saved valid_edt is already local time and is required for interpolated rows.
    if "valid_edt" not in gfs_df.columns:
        if "valid" not in gfs_df.columns:
            raise ValueError("GFS input must contain valid_edt or valid timestamps.")
        # Raw GFS valid timestamps are UTC-naive.
        gfs_df["valid_edt"] = (
            pd.to_datetime(gfs_df["valid"], utc=True)
            .dt.tz_convert("America/New_York")
            .dt.tz_localize(None)
        )
    else:
        gfs_df["valid_edt"] = pd.to_datetime(gfs_df["valid_edt"])

    if "valid" in gfs_df.columns:
        valid_utc = pd.to_datetime(
            gfs_df["valid"], errors="coerce", utc=True, format="mixed"
        )
        expected_valid_edt = (
            valid_utc.dt.tz_convert("America/New_York").dt.tz_localize(None)
        )
        mismatched_valid = (
            gfs_df["valid_edt"].notna()
            & expected_valid_edt.notna()
            & gfs_df["valid_edt"].ne(expected_valid_edt)
        )
        if mismatched_valid.any():
            raise ValueError(
                "GFS valid_edt does not match valid converted from UTC to Eastern time."
            )

    if "init_valid" not in gfs_df.columns:
        raise ValueError("GFS input must include init_valid to verify the data cutoff.")
    init_utc = pd.to_datetime(
        gfs_df["init_valid"], errors="coerce", utc=True, format="mixed"
    )
    unparseable_init = gfs_df["init_valid"].notna() & init_utc.isna()
    if unparseable_init.any():
        raise ValueError("GFS input contains unparseable init_valid timestamps.")
    init_edt = init_utc.dt.tz_convert("America/New_York").dt.tz_localize(None)
    post_cutoff_init = init_edt.notna() & (init_edt >= CUTOFF_EDT)
    if post_cutoff_init.any():
        print(
            f"Discarding {int(post_cutoff_init.sum())} GFS rows initialized at or "
            "after the cutoff."
        )
        gfs_df = gfs_df.loc[~post_cutoff_init].copy()

    # Keep the latest eligible forecast cycle for each valid hour.
    gfs_df["_init_sort"] = pd.to_datetime(
        gfs_df["init_valid"], errors="coerce", utc=True, format="mixed"
    )

    solar_df["valid_edt"] = pd.to_datetime(solar_df["valid_edt"])

    # Remove duplicates on valid_edt if present
    gfs_df = gfs_df.sort_values(
        ["valid_edt", "_init_sort"], na_position="first", kind="stable"
    ).drop_duplicates("valid_edt", keep="last")
    gfs_df = gfs_df.drop(columns=["_init_sort"], errors="ignore")
    solar_df = solar_df.sort_values("valid_edt").drop_duplicates("valid_edt")
    station_pivoted_df = station_pivoted_df.sort_values(
        "valid_edt"
    ).drop_duplicates("valid_edt")

    # Continuous hourly master timeline
    master_index = pd.date_range(
        start="2024-01-01 00:00:00",
        end="2026-09-30 23:00:00",
        freq="1h",
        name="valid_edt",
    )
    master_df = pd.DataFrame(index=master_index).reset_index()

    # Left join all 3 sources (1 row per hour)
    print("Merging pivoted stations, GFS, and Solar onto continuous timeline...")
    master_df = master_df.merge(station_pivoted_df, on="valid_edt", how="left")
    master_df = master_df.merge(gfs_df, on="valid_edt", how="left")
    master_df = master_df.merge(solar_df, on="valid_edt", how="left")

    if (master_df["valid_edt"] >= CUTOFF_EDT).any():
        future_observations = [
            c
            for c in master_df.columns
            if any(c.startswith(f"{stn.lower()}_") for stn in STATION_MAP)
            and not any(token in c for token in ("_lag_",))
        ]
        if master_df.loc[
            master_df["valid_edt"] >= CUTOFF_EDT, future_observations
        ].notna().any().any():
            raise ValueError("Post-cutoff station observations remain after source filtering.")

    # Linearly interpolate 3-hour GFS gap nulls with slope
    gfs_cols = [c for c in master_df.columns if c.startswith("gfs_")]
    print(f"Interpolating GFS 3-hour gaps across {len(gfs_cols)} columns...")
    master_df[gfs_cols] = master_df[gfs_cols].interpolate(
        method="linear", limit_area="inside"
    )

    target_hours = master_df["valid_edt"].between(CUTOFF_EDT, INFERENCE_END_EDT)
    missing_target_gfs = master_df.loc[
        target_hours & master_df["gfs_tmpf"].isna(), "valid_edt"
    ]
    if not missing_target_gfs.empty:
        missing_times = missing_target_gfs.dt.strftime("%Y-%m-%d %H:%M").tolist()
        print(
            f"WARNING: GFS temperature is missing for "
            f"{len(missing_target_gfs)} inference hours; retaining those rows "
            f"with null GFS features: {missing_times[:5]}"
            + (" ..." if len(missing_times) > 5 else "")
        )

    return master_df


def engineer_spatial_lags_and_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineers multi-station spatial/temporal lags and interaction features."""
    print("Engineering multi-station spatial lag features...")
    df = df.sort_values("valid_edt").reset_index(drop=True)

    # Discover all station temperature columns present (e.g. rdu_tmpf, gso_tmpf, igx_tmpf)
    stn_temp_cols = [
        c
        for c in df.columns
        if c.endswith("_tmpf") and not c.startswith("gfs_")
    ]

    # 1. Multi-Station Lags (1h, 2h, 3h, 6h, 12h, 24h, 48h)
    # Allows model to learn weather movement (e.g., GSO temp 3h ago -> RDU temp now)
    for col in stn_temp_cols:
        for lag in [1, 2, 3, 6, 12, 24, 48]:
            df[f"{col}_lag_{lag}h"] = df[col].shift(lag)

    # 2. Regional Spatial Gradients relative to RDU
    if "rdu_tmpf" in df.columns:
        for col in [c for c in stn_temp_cols if c != "rdu_tmpf"]:
            stn_name = col.split("_")[0]
            df[f"grad_{stn_name}_minus_rdu_tmpf"] = df[col] - df["rdu_tmpf"]

    # 3. GFS Physical Interaction Features
    if "gfs_tmpf" in df.columns and "gfs_dwpf" in df.columns:
        df["gfs_dewpoint_depression"] = df["gfs_tmpf"] - df["gfs_dwpf"]

    if "clearsky_ghi" in df.columns and "gfs_cloud_cover_pct" in df.columns:
        df["gfs_estimated_ghi"] = df["clearsky_ghi"] * (
            1.0 - (df["gfs_cloud_cover_pct"] / 100.0)
        )

    # 4. Target Label: GFS Forecast Residual Error at RDU
    if "rdu_tmpf" in df.columns and "gfs_tmpf" in df.columns:
        df["gfs_temp_residual"] = df["rdu_tmpf"] - df["gfs_tmpf"]

    return df


def execute_3way_split():
    master_df = load_and_preprocess_master()
    master_df = engineer_spatial_lags_and_features(master_df)

    print("\nExecuting 3-way time-aware split...")

    # Train Set: Jan 1, 2024 to Aug 31, 2026
    train_df = master_df[master_df["valid_edt"] < VAL_START_EDT].copy()

    # Validation Set: Sept 1, 2026 to Sept 16, 2026 23:00 EDT (16-day local score window)
    val_df = master_df[
        (master_df["valid_edt"] >= VAL_START_EDT)
        & (master_df["valid_edt"] < CUTOFF_EDT)
    ].copy()

    # Inference Set: Sept 17, 2026 to Sept 30, 2026 23:00 EDT (336 target evaluation hours)
    inference_df = master_df[
        (master_df["valid_edt"] >= CUTOFF_EDT)
        & (master_df["valid_edt"] <= INFERENCE_END_EDT)
    ].copy()

    if len(inference_df) != 336:
        raise ValueError(f"Expected 336 inference hours, got {len(inference_df)}.")
    if len(val_df) != 384:
        raise ValueError(f"Expected 384 validation hours, got {len(val_df)}.")
    if train_df["valid_edt"].ge(VAL_START_EDT).any():
        raise ValueError("Training split includes rows at or after validation start.")
    if val_df["valid_edt"].ge(CUTOFF_EDT).any():
        raise ValueError("Validation split includes rows at or after the cutoff.")
    if inference_df["valid_edt"].lt(CUTOFF_EDT).any():
        raise ValueError("Inference split includes rows before the cutoff.")

    # Mask current-time observations and observation-derived features; retain historical lags.
    obs_target_cols = [
        c
        for c in inference_df.columns
        if (
            any(c.startswith(f"{stn.lower()}_") for stn in STATION_MAP)
            and "_lag_" not in c
        )
        or c.startswith("grad_")
    ]
    obs_target_cols.append("gfs_temp_residual")

    for col in obs_target_cols:
        if col in inference_df.columns:
            inference_df[col] = np.nan

    # Save to disk
    train_df.to_csv(TRAIN_OUTPUT_PATH, index=False)
    val_df.to_csv(VAL_OUTPUT_PATH, index=False)
    inference_df.to_csv(INFERENCE_OUTPUT_PATH, index=False)

    print("\n========================================================")
    print("         SUCCESSFUL MULTI-STATION 3-WAY SPLIT           ")
    print("========================================================")
    print(f"Total Master Columns: {len(master_df.columns)}")
    print(
        f"Station Columns     : {[c for c in master_df.columns if '_tmpf' in c]}"
    )
    print("--------------------------------------------------------")
    print(f"1. Training Set   : {len(train_df):,} rows (1 row/hour)")
    print(
        f"   Window         : {train_df['valid_edt'].min()} -> {train_df['valid_edt'].max()}"
    )
    print(
        f"2. Validation Set : {len(val_df):,} rows ({len(val_df)/24:.1f} days)"
    )
    print(
        f"   Window         : {val_df['valid_edt'].min()} -> {val_df['valid_edt'].max()}"
    )
    print(
        f"3. Inference Set  : {len(inference_df):,} rows ({len(inference_df)/24:.1f} days)"
    )
    print(
        f"   Window         : {inference_df['valid_edt'].min()} -> {inference_df['valid_edt'].max()}"
    )
    print("--------------------------------------------------------")
    print(f"Saved Train CSV     : {TRAIN_OUTPUT_PATH}")
    print(f"Saved Validation CSV: {VAL_OUTPUT_PATH}")
    print(f"Saved Inference CSV : {INFERENCE_OUTPUT_PATH}")
    print("========================================================")


if __name__ == "__main__":
    os.makedirs("Data", exist_ok=True)
    execute_3way_split()