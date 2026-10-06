import numpy as np
import pandas as pd
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "data" / "processed" / "edsa_training_data.csv"
rng = np.random.default_rng(42)

n = 5000
timestamps = pd.date_range("2021-01-01", periods=n, freq="h")
segments = rng.choice(["EDSA-North", "EDSA-Center", "EDSA-South"], n)
directions = rng.choice(["NB", "SB"], n)
hour = timestamps.hour
dow = timestamps.dayofweek
weekend = (dow >= 5).astype(int)

rush = (((hour >= 6) & (hour <= 9)) | ((hour >= 16) & (hour <= 20))).astype(int)
speed = 35 - 13*rush - 2*weekend + rng.normal(0, 3, n)
speed = np.clip(speed, 5, 60)

volume = 900 + 950*rush + 250*(~(weekend.astype(bool))) + rng.normal(0, 180, n)
volume = np.clip(volume, 100, None)

density = 15 + 1.15*(volume/100) + 0.7*rush + rng.normal(0, 2, n)
density = np.clip(density, 5, None)

segment_length = rng.choice([1.0, 1.5, 2.0], n)
travel_time = (segment_length / speed) * 60 + rng.normal(0, 0.25, n)
travel_time = np.clip(travel_time, 0.5, None)

df = pd.DataFrame({
    "timestamp": timestamps,
    "segment": segments,
    "direction": directions,
    "average_speed_kmh": speed.round(3),
    "traffic_volume": volume.round(0),
    "traffic_density": density.round(3),
    "segment_length_km": segment_length,
    "travel_time_minutes": travel_time.round(3),
})
df.to_csv(OUT, index=False)
print(f"Created SYNTHETIC demo dataset: {OUT}")
print(df.head(10).to_string(index=False))
