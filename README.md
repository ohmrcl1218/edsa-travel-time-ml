# EDSA Travel Time ML Machine

A machine learning pipeline that predicts travel time along EDSA segments. It trains and compares three regression models, saves them for reuse, and provides both a command-line prediction script and a Streamlit web interface.

> **Important:** The dataset bundled with this project is **synthetic demo data** generated only to test the software. **Do not use its accuracy or results in any research paper.** Replace it with the real EDSA/MMDA dataset and retrain before reporting any results.

---

## Features

- **Three regression models** (scikit-learn):
  - Linear Regression
  - Decision Tree Regressor
  - Random Forest Regressor
- **Chronological 80/20 train-test split** (earlier data for training, later data for testing, which avoids leaking future information into training)
- **Evaluation metrics:** MAE, RMSE, and R²
- **Preprocessing** for CSV and Excel input files
- **One-hot encoding** of EDSA segment and direction
- **Saved models** in `.joblib` format
- **Prediction script** for command-line use
- **Streamlit app** for interactive predictions in the browser
- **Synthetic demo-data generator** for testing the pipeline

---

## Project Structure

```
edsa-travel-time-ml/
├── app/
│   └── app.py                  # Streamlit prediction interface
├── src/
│   ├── generate_demo_data.py   # Creates synthetic demo dataset
│   ├── train.py                # Preprocesses data, trains and evaluates all models
│   └── predict.py              # Command-line prediction script
├── data/                       # Input datasets (CSV / Excel)
├── models/                     # Saved .joblib models
├── requirements.txt
└── README.md
```

> Adjust folder and file names above if your extracted ZIP differs slightly.

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

### 1. Generate demo data (temporary)

```bash
python src/generate_demo_data.py
```

### 2. Train all three models

```bash
cd src
python train.py
```

This preprocesses the data, performs the chronological 80/20 split, trains Linear Regression, Decision Tree, and Random Forest models, prints MAE / RMSE / R² for each, and saves the trained models as `.joblib` files.

### 3. Launch the prediction interface

```bash
cd ..
streamlit run app/app.py
```

The app opens in your browser, where you can enter inputs (such as EDSA segment, direction, and time) and get predicted travel times.

### 4. Predict from the command line (optional)

```bash
cd src
python predict.py
```

---

## Evaluation Metrics

| Metric | Meaning |
|--------|---------|
| **MAE** | Mean Absolute Error: average size of prediction errors, in the same unit as travel time |
| **RMSE** | Root Mean Squared Error: like MAE but penalizes large errors more |
| **R²** | Proportion of variance in travel time explained by the model (closer to 1 is better) |

---

## Using the Real EDSA/MMDA Dataset

1. Place the real dataset (CSV or Excel) in the `data/` folder, replacing the synthetic demo file.
2. If the actual column names or formats differ from the demo data, update the column mapping in the preprocessing section of `src/train.py`. The project is structured so that only the preprocessing needs adapting.
3. Retrain:
   ```bash
   cd src
   python train.py
   ```
4. Relaunch the Streamlit app to use the newly trained models.

The training, evaluation, and prediction code stays the same.

---

## Notes and Limitations

- Results from the synthetic demo data are **not valid** for research or real-world conclusions.
- Because the split is chronological, the dataset must include a usable date/time column.
- Categorical features (EDSA segment, direction) are one-hot encoded; new segment names in future data require retraining.

---

## License

Add your preferred license here (e.g., MIT) or remove this section.
