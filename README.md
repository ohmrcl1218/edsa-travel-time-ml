# EDSA Travel Time Prediction ML Machine

This project trains three regression models:
1. Linear Regression
2. Decision Tree Regressor
3. Random Forest Regressor

Target:
- travel_time_minutes

Expected raw columns:
- timestamp
- segment
- direction
- average_speed_kmh
- traffic_volume
- traffic_density
- travel_time_minutes

IMPORTANT:
The included demo-data generator creates SYNTHETIC data only for testing the software.
Do not use its model results as empirical research results.

## Run

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python src/generate_demo_data.py
python src/train.py
streamlit run app/app.py
```

Replace `data/processed/edsa_training_data.csv` with your real MMDA/EDSA dataset when available, keeping or mapping the required columns.
