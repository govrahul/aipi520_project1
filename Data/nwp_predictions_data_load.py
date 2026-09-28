import concurrent.futures
import os
import warnings
import numpy as np
import pandas as pd
import xarray as xr
from herbie import Herbie
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

RDU_LAT = 35.8776
RDU_LON = -78.7875

SEARCH_PATTERN = ":(TMP|DPT):2 m|:TCDC:entire atmosphere|:(UGRD|VGRD):10 m:"


def extract_single_step(init_str: str, fxx: int) -> dict | None:
    """Worker task: Fetches a single GFS forecast hour for RDU."""
    try:
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
    except Exception:
        return None


def fetch_parallel_gfs(
    tasks: list[tuple[str, int]], max_workers: int = 8
) -> pd.DataFrame:
    """Executes Herbie GFS tasks concurrently across multiple threads."""
    results = []
    total = len(tasks)
    completed = 0

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

    df = pd.DataFrame(results)
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

    # 2. Build Test Target Evaluation Queue (Sept 16 18Z run, 1-hour steps F06 to F342)
    test_init = "2026-09-16 18:00"
    test_tasks = [(test_init, fxx) for fxx in range(6, 343, 1)]

    # Run Parallel Extractions
    print("--- 1/2: Extracting Historical Training Series ---")
    hist_df = fetch_parallel_gfs(hist_tasks, max_workers=8)

    print("\n--- 2/2: Extracting Evaluation Window Test Run ---")
    test_df = fetch_parallel_gfs(test_tasks, max_workers=8)

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
    combined_df = combined_df.sort_values(by="valid_edt").drop_duplicates(
        subset=["valid_edt"], keep="last"
    )

    full_time_index = pd.date_range(
        start=combined_df["valid_edt"].min(),
        end=combined_df["valid_edt"].max(),
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

    output_path = "Data/gfs_rdu_historical_and_forecast_2024_2026.csv"
    combined_df.to_csv(output_path, index=False)
    print(f"\n--- DONE! Saved interpolated GFS features to {output_path} ---")