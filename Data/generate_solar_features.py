import os
import numpy as np
import pandas as pd
import pvlib

# RDU Airport Coordinates
RDU_LAT = 35.8776
RDU_LON = -78.7875


def generate_rdu_solar_features(
    start_date: str = "2024-01-01 00:00",
    end_date: str = "2026-09-30 23:00",
) -> pd.DataFrame:
    """Generates precise, deterministic hourly solar features for RDU coordinates."""
    print(f"Generating hourly solar position features from {start_date} to {end_date}...")

    # 1. Create a complete hourly datetime index in local Eastern Time
    times = pd.date_range(
        start=start_date, end=end_date, freq="1h", tz="America/New_York"
    )

    # 2. Define Location object
    loc = pvlib.location.Location(RDU_LAT, RDU_LON, tz="America/New_York")

    # 3. Calculate exact solar position
    solar_pos = loc.get_solarposition(times)

    # 4. Calculate extraterrestrial solar irradiance (top-of-atmosphere solar flux)
    dni_extra = pvlib.irradiance.get_extra_radiation(times)

    # 5. Assemble clean DataFrame
    df = pd.DataFrame({
        "valid_edt": times.tz_localize(None),  # Remove tz offset for clean CSV joins
        "solar_zenith": solar_pos["zenith"].round(2).values,
        "solar_elevation": solar_pos["elevation"].round(2).values,
        "solar_azimuth": solar_pos["azimuth"].round(2).values,
        "is_daylight": (solar_pos["elevation"] > 0).astype(int).values,
        "toa_irradiance": dni_extra.round(2).values,
    })

    # 6. Compute Clear-Sky Potential Irradiance (Ineichen via Location API)
    clearsky = loc.get_clearsky(times, model="ineichen")
    df["clearsky_ghi"] = clearsky["ghi"].round(2).values  # Global Horizontal Irradiance

    # 7. Calculate daily daylight duration in hours
    daily_daylight = (
        times.to_series()
        .groupby(times.date)
        .apply(
            lambda g: (loc.get_solarposition(g)["elevation"] > 0).sum()
        )
    )
    df["daylength_hours"] = df["valid_edt"].dt.date.map(daily_daylight).values

    # 8. Add cyclic harmonic time features (sine/cosine encodings)
    hour_of_day = df["valid_edt"].dt.hour
    day_of_year = df["valid_edt"].dt.dayofyear

    df["sin_hour"] = np.sin(2 * np.pi * hour_of_day / 24.0).round(4)
    df["cos_hour"] = np.cos(2 * np.pi * hour_of_day / 24.0).round(4)
    df["sin_day"] = np.sin(2 * np.pi * day_of_year / 365.25).round(4)
    df["cos_day"] = np.cos(2 * np.pi * day_of_year / 365.25).round(4)

    return df


if __name__ == "__main__":
    os.makedirs("Data", exist_ok=True)

    solar_df = generate_rdu_solar_features()

    print("\n--- Solar Feature Generation Complete ---")
    print(f"Total hourly records generated: {len(solar_df)}")
    print("\nSample records:")
    print(
        solar_df[[
            "valid_edt",
            "solar_elevation",
            "is_daylight",
            "clearsky_ghi",
            "daylength_hours",
        ]].head(10)
    )

    output_path = "Data/rdu_solar_features_2024_2026.csv"
    solar_df.to_csv(output_path, index=False)
    print(f"\nSaved solar features to: {output_path}")