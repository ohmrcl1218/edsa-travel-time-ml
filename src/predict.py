from pathlib import Path
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"

def predict(model_name, row):
    model = joblib.load(MODEL_DIR / f"{model_name}.joblib")
    df = pd.DataFrame([row])
    return float(model.predict(df)[0])
