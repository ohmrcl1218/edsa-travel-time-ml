from pathlib import Path
import pandas as pd

REQUIRED = ["timestamp", "segment", "direction", "travel_time_minutes"]
OPTIONAL_NUMERIC = [
    "average_speed_kmh", "traffic_volume", "traffic_density", "distance_km"
]

COLUMN_ALIASES = {
    "average_speed_kmh": ["average_speed_kmh", "avg_speed_kph", "avg_speed"],
    "traffic_volume": ["traffic_volume", "traffic_volume_veh", "volume"],
    "traffic_density": ["traffic_density", "traffic_density_veh_per_km", "density"],
    "travel_time_minutes": ["travel_time_minutes", "travel_time_min", "travel_time"],
    "distance_km": ["distance_km", "segment_length_km"],
}

PRETRIP_NUMERIC = ["year", "day_of_week", "hour", "is_weekend", "distance_km"]
PRETRIP_CATEGORICAL = ["segment", "direction", "period"]

REALTIME_NUMERIC = ["average_speed_kmh", "traffic_volume", "traffic_density"]
REALTIME_CATEGORICAL = ["congestion_level"]


def _normalize_column_names(df):
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

    for canonical, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in df.columns and canonical not in df.columns:
                df = df.rename(columns={alias: canonical})

    if "day_of_week" in df.columns and not pd.api.types.is_numeric_dtype(df["day_of_week"]):
        day_map = {
            "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
            "friday": 4, "saturday": 5, "sunday": 6
        }
        df["day_of_week"] = (
            df["day_of_week"].astype(str).str.strip().str.lower().map(day_map)
        )

    return df


def load_dataset(path):
    path = Path(path)
    if path.suffix.lower() in [".xlsx", ".xls"]:
        df = pd.read_excel(path)
    elif path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError("Use CSV or Excel input.")

    df = _normalize_column_names(df)

    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(
            "Missing required columns: " + ", ".join(missing)
            + "\nExpected: " + ", ".join(REQUIRED)
        )

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["travel_time_minutes"] = pd.to_numeric(df["travel_time_minutes"], errors="coerce")
    for c in OPTIONAL_NUMERIC:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=REQUIRED).copy()
    df = df[df["travel_time_minutes"] > 0]
    if "average_speed_kmh" in df.columns:
        df = df[df["average_speed_kmh"] > 0]
    df = df.sort_values("timestamp").reset_index(drop=True)

    df["year"] = df["timestamp"].dt.year
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["hour"] = df["timestamp"].dt.hour
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    if "period" in df.columns:
        df["period"] = df["period"].astype(str).str.upper().str.strip()

    return df
