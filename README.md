# EDSA Travel Time ML Machine

A machine learning pipeline that predicts travel time along EDSA segments. It trains and compares three regression models, saves them for reuse, and provides both a command-line prediction script and a Streamlit web interface.

---

## Features

- **Three regression models** (scikit-learn):
  - Linear Regression
  - Decision Tree Regressor
  - Random Forest Regressor
- **Chronological 80/20 train-test split** (earlier data for training, later data for testing, which avoids leaking future information into training)
- **Evaluation metrics:** MAE, RMSE, and R²
- **Preprocessing** for CSV and Excel input files
- **Pre-trip features only** (no data leakage, see below)
- **One-hot encoding** of EDSA segment, direction and period
- **Saved models** in `.joblib` format
- **Prediction script** for command-line use
- **Streamlit app** for interactive predictions in the browser

---

## Project Structure

```
edsa-travel-time-ml/
├── app/
│   └── app.py                  # Streamlit prediction interface
├── src/
│   ├── preprocess.py           # Loading, cleaning, feature lists
│   ├── train.py                # Trains and evaluates all models + leakage check
│   ├── predict.py              # Command-line prediction script
│   └── analyze_relationships.py# Correlation / ANOVA analysis and graphs
├── data/processed/             # Input dataset (CSV / Excel)
├── models/                     # Saved .joblib models, metrics, metadata.json
├── results/                    # Correlation tables, graphs, leakage_comparison.csv
├── requirements.txt
└── README.md
```

---

## Requirements

- Python 3.9 or newer
- Packages listed in `requirements.txt` (includes scikit-learn, pandas, joblib, Streamlit, and Excel readers)

---

## Installation

Extract the ZIP, open Command Prompt inside the project folder, then run:

```bash
python -m venv .venv
```

Activate the virtual environment (Windows):

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

### 1. Train all three models

```bash
cd src
python train.py
```

This preprocesses the data, performs the chronological 80/20 split, trains Linear Regression, Decision Tree, and Random Forest models, prints MAE / RMSE / R² for each, and saves the trained models as `.joblib` files.

### 2. Launch the prediction interface

```bash
cd ..
streamlit run app/app.py
```

The app opens in your browser, where you can enter inputs (EDSA segment, direction, year, day of week, and departure hour) and get predicted travel times.

### 3. Predict from the command line (optional)

```bash
cd src
python predict.py
```

---

## Features and Data Leakage

The models are meant to help commuters estimate travel time **before they leave**, so they only use information that is known at that moment:

| Used (known before the trip) | Excluded (measured during the trip) |
|---|---|
| year, day of week, hour, weekend flag, period (AM/OFF/PM), segment, direction, segment distance | average speed, traffic volume, traffic density, congestion level |

**Why speed is excluded:** speed is just `distance / travel time`, so a model that sees it can recover the answer directly instead of learning traffic patterns. Scores look excellent (R² close to 1) but are not realistic, and a commuter does not know the trip's average speed in advance. Volume, density and congestion level are likewise only known from conditions during the trip.

`python train.py` also writes `results/leakage_comparison.csv`, which compares the pre-trip models with the same models given the real-time variables. This is for reference in the paper only; those models are not saved or used by the app.

Speed, volume and density are still used in `analyze_relationships.py` for the correlation analysis (relationship with travel time), which is descriptive and not a prediction.

`year` is included because travel time on this dataset rises steadily from 2021 to 2025, and the chronological split tests on the latest records. Without it the models cannot account for that trend.

---

## Evaluation Metrics

| Metric | Meaning |
|--------|---------|
| **MAE** | Mean Absolute Error: average size of prediction errors, in the same unit as travel time |
| **RMSE** | Root Mean Squared Error: like MAE but penalizes large errors more |
| **R²** | Proportion of variance in travel time explained by the model (closer to 1 is better) |

---

## Notes and Limitations

- Predictions are only as reliable as the data: the dataset covers a limited set of segments and departure hours, and the app warns when an hour is outside the training data.
- The model uses the date to account for the yearly trend, so predictions for years beyond the data are extrapolations.
- Because the split is chronological, the dataset must include a usable date/time column.
- Categorical features (segment, direction, period) are one-hot encoded; new segment names in future data require retraining.
