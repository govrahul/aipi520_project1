| Column Name | Parameter Name | Description & Standard Unit |
| :--- | :--- | :--- |
| `station` | Station Identifier | The 3- or 4-character ICAO/FAA station code (e.g., `RDU`, `GSO`, `FAY`). |
| `valid` | Observation Timestamp | Timestamp of the reading in local time (`America/New_York`), formatted as `YYYY-MM-DD HH:MM:SS`. |
| `lon` | Longitude | Geographic longitude coordinate of the weather station in decimal degrees. |
| `lat` | Latitude | Geographic latitude coordinate of the weather station in decimal degrees. |
| `tmpf` | Air Temperature | Dry-bulb surface air temperature measured in **Degrees Fahrenheit (°F)**. |
| `dwpf` | Dew Point Temperature | Dew point temperature measured in **Degrees Fahrenheit (°F)**. (Higher dew points indicate greater atmospheric moisture/humidity). |
| `relh` | Relative Humidity | Relative humidity expressed as a **percentage (%)**. |
| `drct` | Wind Direction | Meteorological direction from which the wind is blowing, measured in **Degrees from True North** (0°–360°). 0°/360° represents North, 90° East, 180° South, and 270° West. |
| `sknt` | Wind Speed | Sustained wind speed measured in **Knots** (1 knot ≈ 1.151 mph). |
| `p01i` | Precipitation Accumulation | Total precipitation accumulated during the preceding 1-hour interval, measured in **Inches (in)**. |
| `alti` | Altimeter Setting | Barometric pressure setting corrected to sea level, measured in **Inches of Mercury (inHg)** (standard sea level pressure is ~29.92 inHg). |
| `mslp` | Mean Sea Level Pressure | Atmospheric pressure recalculated to mean sea level, measured in **Millibars / Hectopascals (mb / hPa)** (standard sea level pressure is ~1013.25 mb). |