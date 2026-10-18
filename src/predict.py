from pathlib import Path
import json
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"


def load_metadata():
    return json.loads((MODEL_DIR / "metadata.json").read_text())


def predict(model_name, row):
    """row: dict with the pre-trip features listed in models/metadata.json."""
    model = joblib.load(MODEL_DIR / f"{model_name}.joblib")
    features = load_metadata()["features"]
    df = pd.DataFrame([row])[features]
    return float(model.predict(df)[0])


if __name__ == "__main__":
    meta = load_metadata()
    segment = meta["segments"][0]
    example = {
        "year": 2025,
        "day_of_week": 1,
        "hour": 7,
        "is_weekend": 0,
        "distance_km": meta["segment_distance_km"].get(segment, 23.08),
        "segment": segment,
        "direction": meta["directions"][0],
        "period": meta["hour_to_period"].get("7", "AM"),
    }
    print("Input:", example)
    for name in ["linear_regression", "decision_tree", "random_forest"]:
        print(f"{name}: {predict(name, example):.2f} minutes")
