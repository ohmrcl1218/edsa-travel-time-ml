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

from preprocess import load_dataset

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "edsa_training_data.csv"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

df = load_dataset(DATA)

# Chronological split: earliest 80% for training, latest 20% for testing.
split = int(len(df) * 0.80)
train = df.iloc[:split].copy()
test = df.iloc[split:].copy()

features = [
    "day_of_week", "hour", "is_weekend",
    "average_speed_kmh", "traffic_volume",
    "traffic_density", "segment", "direction"
]
target = "travel_time_minutes"

X_train, y_train = train[features], train[target]
X_test, y_test = test[features], test[target]

numeric = [
    "day_of_week", "hour", "is_weekend",
    "average_speed_kmh", "traffic_volume", "traffic_density"
]
categorical = ["segment", "direction"]

preprocessor = ColumnTransformer([
    ("num", "passthrough", numeric),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical)
])

models = {
    "linear_regression": LinearRegression(),
    "decision_tree": DecisionTreeRegressor(
        max_depth=12, min_samples_leaf=5, random_state=42
    ),
    "random_forest": RandomForestRegressor(
        n_estimators=300, max_depth=18, min_samples_leaf=2,
        random_state=42, n_jobs=-1
    ),
}

results = []

for name, estimator in models.items():
    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", estimator)
    ])

    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)

    rmse = mean_squared_error(y_test, pred) ** 0.5
    mae = mean_absolute_error(y_test, pred)
    r2 = r2_score(y_test, pred)

    joblib.dump(pipe, MODEL_DIR / f"{name}.joblib")

    results.append({
        "model": name,
        "MAE_minutes": round(mae, 4),
        "RMSE_minutes": round(rmse, 4),
        "R2": round(r2, 4)
    })

results_df = pd.DataFrame(results)
results_df.to_csv(MODEL_DIR / "model_results.csv", index=False)

metadata = {
    "target": target,
    "features": features,
    "train_rows": len(train),
    "test_rows": len(test),
    "split": "chronological 80/20",
    "WARNING": "If the input is synthetic demo data, results are software-test results, not research results."
}
(MODEL_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2))

print("\nMODEL RESULTS")
print(results_df.to_string(index=False))
print(f"\nSaved models to: {MODEL_DIR}")
