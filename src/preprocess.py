from pathlib import Path
import pandas as pd

REQUIRED = [
    "timestamp", "segment", "direction",
    "average_speed_kmh", "traffic_volume",
    "traffic_density", "travel_time_minutes"
]

def load_dataset(path):
    path = Path(path)
    if path.suffix.lower() in [".xlsx", ".xls"]:
        df = pd.read_excel(path)
    elif path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError("Use CSV or Excel input.")

    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(
            "Missing required columns: " + ", ".join(missing)
            + "\nExpected: " + ", ".join(REQUIRED)
        )

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    for c in ["average_speed_kmh", "traffic_volume", "traffic_density", "travel_time_minutes"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=REQUIRED).copy()
    df = df[df["average_speed_kmh"] > 0]
    df = df[df["travel_time_minutes"] > 0]
    df = df.sort_values("timestamp").reset_index(drop=True)

    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["hour"] = df["timestamp"].dt.hour
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    return df

def make_features(df):
    X = df[
        [
            "day_of_week", "hour", "is_weekend",
            "average_speed_kmh", "traffic_volume",
            "traffic_density"
        ]
    ].copy()

    X = pd.get_dummies(
        pd.concat([X, df[["segment", "direction"]]], axis=1),
        columns=["segment", "direction"],
        dtype=int
    )
    return X
