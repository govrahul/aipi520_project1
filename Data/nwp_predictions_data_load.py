import concurrent.futures
import os
import time
import warnings
import numpy as np
import pandas as pd
import requests
import xarray as xr
from herbie import Herbie
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

RDU_LAT = 35.8776
RDU_LON = -78.7875

CUTOFF_EDT = pd.Timestamp("2026-09-17 00:00", tz="America/New_York")
TARGET_START_EDT = pd.Timestamp("2026-09-17 00:00", tz="America/New_York")
TARGET_END_EDT = pd.Timestamp("2026-09-30 23:00", tz="America/New_York")
GFS_TEST_INIT_UTC = pd.Timestamp("2026-09-16 18:00", tz="UTC")
MAX_EXTRACTION_ATTEMPTS = 3
RETRY_BASE_DELAY_SECONDS = 2

SEARCH_PATTERN = ":(TMP|DPT):2 m|:TCDC:entire atmosphere|:(UGRD|VGRD):10 m:"


def _extract_single_step_once(init_str: str, fxx: int) -> dict:
    """Fetch and process one GFS forecast hour."""
    H = Herbie(
        init_str,
        model="gfs",
        product="pgrb2.0p25",
        fxx=fxx,
        verbose=False,
    )
    ds = H.xarray(SEARCH_PATTERN, remove_grib=True)

    if isinstance(ds, list):
        ds = xr.merge(ds, compat="override")

    gfs_lon = RDU_LON + 360 if RDU_LON < 0 else RDU_LON

    # Fast xarray point indexing (bypasses cartopy)
    if "latitude" in ds.coords and "longitude" in ds.coords:
        point = ds.sel(latitude=RDU_LAT, longitude=gfs_lon, method="nearest")
    elif "lat" in ds.coords and "lon" in ds.coords:
        point = ds.sel(lat=RDU_LAT, lon=gfs_lon, method="nearest")
    else:
        point = ds.herbie.nearest_points(points=(gfs_lon, RDU_LAT))

    valid_time = pd.to_datetime(point["valid_time"].values).tz_localize(None)

    tmp_k = float(point["t2m"].values) if "t2m" in point else np.nan
    dpt_k = float(point["d2m"].values) if "d2m" in point else np.nan

    tmpf = (tmp_k - 273.15) * 9 / 5 + 32 if not np.isnan(tmp_k) else np.nan
    dwpf = (dpt_k - 273.15) * 9 / 5 + 32 if not np.isnan(dpt_k) else np.nan

    u10 = float(point["u10"].values) if "u10" in point else np.nan
    v10 = float(point["v10"].values) if "v10" in point else np.nan
    wind_speed_m_s = np.sqrt(u10**2 + v10**2) if not np.isnan(u10) else np.nan
    wind_speed_knots = (
        wind_speed_m_s * 1.94384 if not np.isnan(wind_speed_m_s) else np.nan
    )

    tcdc = float(point["tcdc"].values) if "tcdc" in point else np.nan

    return {
        "init_valid": pd.to_datetime(init_str),
        "forecast_hour": fxx,
        "valid": valid_time,
        "gfs_tmpf": round(tmpf, 2),
        "gfs_dwpf": round(dwpf, 2),
        "gfs_sknt": round(wind_speed_knots, 2),
        "gfs_cloud_cover_pct": round(tcdc, 2),
    }


def extract_single_step(init_str: str, fxx: int) -> dict | None:
    """Fetch one forecast step, retrying transient network errors with backoff."""
    retryable_messages = (
        "timed out",
        "timeout",
        "connection reset",
        "connection aborted",
        "temporary failure",
        "temporarily unavailable",
    )

    for attempt in range(1, MAX_EXTRACTION_ATTEMPTS + 1):
        try:
            return _extract_single_step_once(init_str, fxx)
        except Exception as exc:
            is_network_error = isinstance(exc, requests.exceptions.RequestException)
            is_transient = is_network_error or any(
                message in str(exc).lower() for message in retryable_messages
            )
            if not is_transient or attempt == MAX_EXTRACTION_ATTEMPTS:
                print(
                    f"Failed to extract GFS init={init_str}, f{fxx:03d} "
                    f"after {attempt} attempt(s): {exc}"
                )
                return None

            delay = RETRY_BASE_DELAY_SECONDS * (2 ** (attempt - 1))
            print(
                f"Transient GFS error for init={init_str}, f{fxx:03d}; "
                f"retrying in {delay}s (attempt {attempt + 1}/"
                f"{MAX_EXTRACTION_ATTEMPTS}): {exc}"
            )
            time.sleep(delay)

    return None


def fetch_parallel_gfs(
    tasks: list[tuple[str, int]], max_workers: int = 8
) -> pd.DataFrame:
    """Executes Herbie GFS tasks concurrently across multiple threads."""
    results = []
    failed_tasks = []
    total = len(tasks)

    print(f"Starting parallel download for {total} forecast steps ({max_workers} threads)...")

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {
            executor.submit(extract_single_step, init_str, fxx): (init_str, fxx)
            for init_str, fxx in tasks
        }

        for future in tqdm(
            concurrent.futures.as_completed(future_to_task),
            total=total,
            desc="Downloading GFS Steps",
            unit="step",
        ):
            res = future.result()
            if res is not None:
                results.append(res)
            else:
                failed_tasks.append(future_to_task[future])

    if failed_tasks:
        failed_summary = ", ".join(
            f"{init_str} F{fxx:03d}" for init_str, fxx in failed_tasks[:10]
        )
        if len(failed_tasks) > 10:
            failed_summary += f", ... ({len(failed_tasks)} total)"
        print(
            f"WARNING: {len(failed_tasks)} GFS extraction task(s) failed after "
            f"retries; continuing with successful results: {failed_summary}"
        )

    result_columns = [
        "init_valid",
        "forecast_hour",
        "valid",
        "gfs_tmpf",
        "gfs_dwpf",
        "gfs_sknt",
        "gfs_cloud_cover_pct",
    ]
    df = pd.DataFrame(results, columns=result_columns)
    return df


if __name__ == "__main__":
    os.makedirs("Data", exist_ok=True)

    # 1. Build Historical Task Queue (3-hour steps: F03, F06, F09, ..., F24)
    hist_dates = pd.date_range(start="2024-01-01", end="2026-09-16", freq="1D")
    hist_tasks = []
    for dt in hist_dates:
        init_str = dt.strftime("%Y-%m-%d 00:00")
        for fxx in range(3, 25, 3):  # 3-hour stride
            hist_tasks.append((init_str, fxx))

    # Build a forecast queue matching the complete local inference window.
    test_init = GFS_TEST_INIT_UTC.strftime("%Y-%m-%d %H:%M")
    start_fxx = int(
        (TARGET_START_EDT.tz_convert("UTC") - GFS_TEST_INIT_UTC).total_seconds()
        // 3600
    )
    end_fxx = int(
        (TARGET_END_EDT.tz_convert("UTC") - GFS_TEST_INIT_UTC).total_seconds()
        // 3600
    )
    if GFS_TEST_INIT_UTC >= CUTOFF_EDT.tz_convert("UTC"):
        raise ValueError("GFS test initialization must be before the data cutoff.")
    if start_fxx < 0 or end_fxx < start_fxx:
        raise ValueError("GFS initialization does not cover the requested target window.")
    # Sample the target at 3-hour intervals to reduce downloads. GFS switches
    # to 3-hourly output after F120, so retain that transition and final endpoint.
    last_hourly_fxx = min(end_fxx, 120)
    test_forecast_hours = list(range(start_fxx, last_hourly_fxx + 1, 3))
    if last_hourly_fxx >= 120 and 120 not in test_forecast_hours:
        test_forecast_hours.append(120)
    if end_fxx > 120:
        test_forecast_hours.extend(range(123, end_fxx + 1, 3))
    elif end_fxx not in test_forecast_hours:
        test_forecast_hours.append(end_fxx)
    test_tasks = [(test_init, fxx) for fxx in test_forecast_hours]

    # Run Parallel Extractions
    print("--- 1/2: Extracting Historical Training Series ---")
    hist_df = fetch_parallel_gfs(hist_tasks, max_workers=8)

    print("\n--- 2/2: Extracting Evaluation Window Test Run ---")
    test_df = fetch_parallel_gfs(test_tasks, max_workers=8)
    returned_fxx = set(pd.to_numeric(test_df["forecast_hour"], errors="coerce").dropna().astype(int))
    expected_fxx = set(test_forecast_hours)
    missing_fxx = sorted(expected_fxx - returned_fxx)
    unexpected_fxx = sorted(returned_fxx - expected_fxx)
    if missing_fxx or unexpected_fxx or len(test_df) != len(expected_fxx):
        print(
            "WARNING: GFS extraction returned a partial 3-hourly sample. "
            f"Missing forecast hours: {missing_fxx[:10]}"
            + (" ..." if len(missing_fxx) > 10 else "")
            + f"; unexpected forecast hours: {unexpected_fxx[:10]}. "
            f"Continuing with {len(test_df)} successful forecast steps."
        )
    expected_valid = pd.to_datetime(
        test_df["init_valid"], utc=True, format="mixed"
    ) + pd.to_timedelta(test_df["forecast_hour"].astype(int), unit="h")
    actual_valid = pd.to_datetime(test_df["valid"], utc=True, format="mixed")
    timestamp_mismatches = (
        actual_valid.dt.tz_convert("UTC").dt.tz_localize(None).to_numpy(
            dtype="datetime64[ns]"
        )
        != expected_valid.dt.tz_convert("UTC").dt.tz_localize(None).to_numpy(
            dtype="datetime64[ns]"
        )
    )
    if timestamp_mismatches.any():
        mismatch_rows = test_df.loc[
            timestamp_mismatches, ["init_valid", "forecast_hour", "valid"]
        ].head(5)
        print(
            "WARNING: Some returned GFS timestamps differ from init + forecast "
            f"hour; keeping their extracted valid_time values. Examples: "
            f"{mismatch_rows.to_dict(orient='records')}"
        )
    missing_temps = int(test_df["gfs_tmpf"].isna().sum())
    if missing_temps:
        print(
            f"WARNING: {missing_temps} returned GFS forecast step(s) have no "
            "temperature; retaining them as missing."
        )

    # Combine datasets
    combined_df = pd.concat([hist_df, test_df], ignore_index=True)

    # Localize to Eastern Time (EDT)
    combined_df["valid_edt"] = (
        pd.to_datetime(combined_df["valid"])
        .dt.tz_localize("UTC")
        .dt.tz_convert("America/New_York")
        .dt.tz_localize(None)
    )

    # Sort and resample/interpolate 3-hour historical steps back to clean 1-hour intervals
    combined_df = combined_df.sort_values(
        by=["valid_edt", "init_valid", "forecast_hour"],
        na_position="first",
        kind="stable",
    ).drop_duplicates(subset=["valid_edt"], keep="last")

    full_time_index = pd.date_range(
        start=combined_df["valid_edt"].min(),
        end=max(
            combined_df["valid_edt"].max(),
            TARGET_END_EDT.tz_localize(None),
        ),
        freq="1h",
    )
    combined_df = (
        combined_df.set_index("valid_edt")
        .reindex(full_time_index)
        .rename_axis("valid_edt")
        .reset_index()
    )

    # Linearly interpolate missing 1-hour steps from the 3-hour stride
    numeric_cols = ["gfs_tmpf", "gfs_dwpf", "gfs_sknt", "gfs_cloud_cover_pct"]
    combined_df[numeric_cols] = combined_df[numeric_cols].interpolate(
        method="linear"
    )

    target_df = combined_df[
        combined_df["valid_edt"].between(
            TARGET_START_EDT.tz_localize(None), TARGET_END_EDT.tz_localize(None)
        )
    ]
    expected_target_index = pd.date_range(
        TARGET_START_EDT.tz_localize(None),
        TARGET_END_EDT.tz_localize(None),
        freq="1h",
    )
    missing_target_times = expected_target_index.difference(target_df["valid_edt"])
    missing_target_temps = int(target_df["gfs_tmpf"].isna().sum())
    if len(missing_target_times) or missing_target_temps:
        print(
            "WARNING: Writing partial GFS target coverage: "
            f"{len(expected_target_index) - len(missing_target_times)}/"
            f"{len(expected_target_index)} target hours are present; "
            f"{missing_target_temps} have no interpolated temperature."
        )

    output_path = "Data/gfs_rdu_historical_and_forecast_2024_2026.csv"
    combined_df.to_csv(output_path, index=False)
    print(f"\n--- DONE! Saved interpolated GFS features to {output_path} ---")