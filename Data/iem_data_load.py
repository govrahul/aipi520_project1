from io import StringIO
import pandas as pd
import requests

CUTOFF_DATETIME = "2026-09-17 00:00"

STATION_NAMES = {
    "RDU": "Raleigh-Durham International",
    "IGX": "Chapel Hill / Horace Williams",
    "RWI": "Rocky Mount-Wilson",
    "GSO": "Greensboro / High Point",
    "FAY": "Fayetteville Regional",
}

STATIONS = list(STATION_NAMES.keys())


def fetch_all_asos_data(
    stations: list[str], start_date="2024-01-01", end_date="2026-09-17"
) -> pd.DataFrame:
    """Fetches historical hourly ASOS data for multiple stations in a SINGLE request to avoid rate limits."""
    base_url = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"

    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)

    # Build request params
    params = [
        ("data", "tmpf"),
        ("data", "dwpf"),
        ("data", "relh"),
        ("data", "drct"),
        ("data", "sknt"),
        ("data", "p01i"),
        ("data", "alti"),
        ("data", "mslp"),
        ("year1", str(start_dt.year)),
        ("month1", str(start_dt.month)),
        ("day1", str(start_dt.day)),
        ("year2", str(end_dt.year)),
        ("month2", str(end_dt.month)),
        ("day2", str(end_dt.day)),
        ("tz", "America/New_York"),
        ("format", "onlycomma"),
        ("latlon", "yes"),
        ("missing", "M"),
    ]

    # Add each station as a separate query parameter
    for st in stations:
        params.append(("station", st))

    headers = {
        # Polite User-Agent prevents generic bot blocking
        "User-Agent": "AcademicResearchProject/1.0 (Student Project)"
    }

    print(f"Requesting data for stations {stations}...")
    response = requests.get(base_url, params=params, headers=headers)

    if response.status_code == 200:
        df = pd.read_csv(
            StringIO(response.text), 
            comment="#",
            low_memory=False,
            dtype=str
        )
        return df
    else:
        raise RuntimeError(
            f"Failed to fetch data: HTTP Status {response.status_code}"
        )


def clean_and_filter_data(
    df: pd.DataFrame, cutoff_timestamp: str
) -> pd.DataFrame:
    """Standardizes columns, coerces types, and applies strict temporal cutoff."""
    # Standardize column names
    df.columns = [col.strip() for col in df.columns]

    # Parse timestamp
    df["valid"] = pd.to_datetime(df["valid"])

    # Strict cutoff filter (< 2026-09-17 00:00:00)
    cutoff_dt = pd.to_datetime(cutoff_timestamp)
    df = df[df["valid"] < cutoff_dt].copy()

    # Coerce numeric columns
    numeric_cols = [
        "tmpf",
        "dwpf",
        "relh",
        "drct",
        "sknt",
        "p01i",
        "alti",
        "mslp",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Sort chronologically
    df = df.sort_values(by=["station", "valid"]).reset_index(drop=True)
    return df


def resample_to_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """Resamples 5-minute ASOS observations into clean hourly time steps for each station."""
    df = df.sort_values(by=["station", "valid"]).copy()

    # Resample to 1-hour intervals per station averaging sub-hourly readings
    hourly_df = (
        df.set_index("valid")
        .groupby("station")
        .resample("1h")
        .mean(numeric_only=True)
        .reset_index()
    )

    numeric_cols = ["tmpf", "dwpf", "relh", "drct", "sknt", "p01i", "alti", "mslp"]

    # Forward-fill isolated missing hourly gaps (up to 2 consecutive hours)
    hourly_df[numeric_cols] = (
        hourly_df.groupby("station")[numeric_cols]
        .transform(lambda group: group.ffill(limit=2))
    )

    return hourly_df


if __name__ == "__main__":
    # Fetch raw observation data
    raw_df = fetch_all_asos_data(
        STATIONS, start_date="2024-01-01", end_date="2026-09-17"
    )

    # Clean and enforce strict cutoff date
    cleaned_df: pd.DataFrame = clean_and_filter_data(raw_df, CUTOFF_DATETIME)

    # Aggregate 5-minute sub-hourly reports to regular 1-hour resolution
    combined_df: pd.DataFrame = resample_to_hourly(cleaned_df)

    print(f"\nSuccessfully fetched and processed {len(combined_df)} hourly rows!")
    print("\nPreview of hourly data:")
    print(combined_df[["station", "valid", "tmpf", "dwpf", "sknt"]].head(10))

    # Save to CSV
    combined_df.to_csv("Data/station_historical_data.csv", index=False)
    print("\nData saved to Data/station_historical_data.csv")