import sys
from pathlib import Path
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from predict import predict

st.set_page_config(page_title="EDSA Travel Time Predictor", page_icon="🚗")
st.title("EDSA Travel Time Prediction")
st.caption("ML prototype — replace demo data with verified EDSA/MMDA observations before research use.")

segments = ["EDSA-North", "EDSA-Center", "EDSA-South"]
directions = ["NB", "SB"]

segment = st.selectbox("EDSA Segment", segments)
direction = st.selectbox("Direction", directions)
day = st.selectbox(
    "Day of Week",
    list(range(7)),
    format_func=lambda x: ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"][x]
)
hour = st.slider("Hour", 0, 23, 8)
speed = st.number_input("Average Speed (km/h)", min_value=1.0, max_value=100.0, value=18.0)
volume = st.number_input("Traffic Volume (vehicles)", min_value=0.0, value=2500.0)
density = st.number_input("Traffic Density", min_value=0.0, value=75.0)

row = {
    "day_of_week": day,
    "hour": hour,
    "is_weekend": int(day >= 5),
    "average_speed_kmh": speed,
    "traffic_volume": volume,
    "traffic_density": density,
    "segment": segment,
    "direction": direction,
}

if st.button("Predict Travel Time"):
    st.subheader("Predictions")
    for model in ["linear_regression", "decision_tree", "random_forest"]:
        try:
            value = predict(model, row)
            st.write(f"**{model.replace('_', ' ').title()}:** {value:.2f} minutes")
        except FileNotFoundError:
            st.error("Models not found. Run: python src/train.py")
            break
