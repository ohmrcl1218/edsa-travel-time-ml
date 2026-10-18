from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from preprocess import (
    load_dataset,
    PRETRIP_NUMERIC, PRETRIP_CATEGORICAL,
    REALTIME_NUMERIC, REALTIME_CATEGORICAL,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "edsa_training_data.csv"
MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

TARGET = "travel_time_minutes"

df = load_dataset(DATA)

split = int(len(df) * 0.80)
train = df.iloc[:split].copy()
test = df.iloc[split:].copy()


def build_models():
    return {
        "linear_regression": LinearRegression(),
        "decision_tree": DecisionTreeRegressor(
            max_depth=12, min_samples_leaf=5, random_state=42
        ),
        "random_forest": RandomForestRegressor(
            n_estimators=300, max_depth=18, min_samples_leaf=2,
            random_state=42, n_jobs=-1
        ),
    }


def run(numeric, categorical, save=False):
    """Train/evaluate all three models on the given feature lists."""
    numeric = [c for c in numeric if c in df.columns]
    categorical = [c for c in categorical if c in df.columns]
    features = numeric + categorical

    preprocessor = ColumnTransformer([
        ("num", "passthrough", numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ])

    rows = []
    for name, estimator in build_models().items():
        pipe = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
        pipe.fit(train[features], train[TARGET])
        pred = pipe.predict(test[features])

        rows.append({
            "model": name,
            "MAE_minutes": round(mean_absolute_error(test[TARGET], pred), 4),
            "RMSE_minutes": round(mean_squared_error(test[TARGET], pred) ** 0.5, 4),
            "R2": round(r2_score(test[TARGET], pred), 4),
        })
        if save:
            joblib.dump(pipe, MODEL_DIR / f"{name}.joblib")

    return features, pd.DataFrame(rows)


features, results_df = run(PRETRIP_NUMERIC, PRETRIP_CATEGORICAL, save=True)
results_df.to_csv(MODEL_DIR / "model_results.csv", index=False)

_, leaky_df = run(
    PRETRIP_NUMERIC + REALTIME_NUMERIC,
    PRETRIP_CATEGORICAL + REALTIME_CATEGORICAL,
)
comparison = results_df.merge(
    leaky_df, on="model", suffixes=("_pretrip", "_with_realtime")
)
comparison.to_csv(RESULTS_DIR / "leakage_comparison.csv", index=False)

segment_distance = (
    df.groupby("segment")["distance_km"].median().round(2).to_dict()
    if "distance_km" in df.columns else {}
)
hour_period = (
    df.groupby("hour")["period"].agg(lambda s: s.mode().iloc[0]).to_dict()
    if "period" in df.columns else {}
)

metadata = {
    "target": TARGET,
    "features": features,
    "excluded_features": REALTIME_NUMERIC + REALTIME_CATEGORICAL,
    "exclusion_reason": (
        "Measured during the trip, not known beforehand. Speed = distance/time, "
        "so including it leaks the target; volume, density and congestion level "
        "are also unavailable to a commuter before departure."
    ),
    "train_rows": len(train),
    "test_rows": len(test),
    "split": "chronological 80/20",
    "segments": sorted(df["segment"].unique().tolist()),
    "directions": sorted(df["direction"].unique().tolist()),
    "segment_distance_km": segment_distance,
    "hour_to_period": {str(k): v for k, v in hour_period.items()},
}
(MODEL_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2))

print("\nMODEL RESULTS (pre-trip features only)")
print(results_df.to_string(index=False))
print("\nLEAKAGE CHECK (pre-trip vs. with real-time variables)")
print(comparison.to_string(index=False))
print(f"\nSaved models to: {MODEL_DIR}")
