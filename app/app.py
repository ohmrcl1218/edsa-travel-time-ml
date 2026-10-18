import sys
from pathlib import Path
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from predict import predict, load_metadata

st.set_page_config(page_title="EDSA Travel Time Predictor", page_icon="🚗")
st.title("EDSA Travel Time Prediction")
st.caption(
    "Predicts travel time before you leave, using only the segment, direction, "
    "day and time of departure."
)

try:
    meta = load_metadata()
except FileNotFoundError:
    st.error("Models not found. Run: python src/train.py")
    st.stop()

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]

segment = st.selectbox("EDSA Segment", meta["segments"])
direction = st.selectbox("Direction", meta["directions"])
year = st.number_input("Year", min_value=2021, max_value=2035, value=2025, step=1)
day = st.selectbox("Day of Week", list(range(len(DAYS))), format_func=lambda x: DAYS[x])
hour = st.slider("Departure Hour", 0, 23, 7)

period = meta["hour_to_period"].get(str(hour))
if period is None:
    period = "AM" if 6 <= hour < 12 else "PM" if 12 <= hour < 20 else "OFF"
    st.info(
        f"The training data has no records for {hour}:00, so this prediction "
        "is an extrapolation and may be less reliable."
    )

row = {
    "year": int(year),
    "day_of_week": day,
    "hour": hour,
    "is_weekend": int(day >= 5),
    "distance_km": meta["segment_distance_km"].get(segment, 23.08),
    "segment": segment,
    "direction": direction,
    "period": period,
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
