# Istanbul Rain Prediction

A machine learning project that predicts whether it will rain in Istanbul tomorrow,
using historical weather data from the Open-Meteo API.

---

## Problem Statement

Istanbul has a complex climate influenced by the Black Sea, the Marmara Sea, and
the Bosphorus strait. Rainfall is highly seasonal — the October–March period
accounts for the majority of annual precipitation, while summers are largely dry.

This project builds a binary classifier to predict **rain tomorrow** (≥ 1mm
precipitation) from daily weather observations. The target variable is defined
using the standard meteorological threshold of 1mm, which filters out
negligible moisture such as dew.

---

## Data

**Source:** [Open-Meteo Historical Weather API](https://open-meteo.com)
**License:** CC BY 4.0 — free to use, no account required
**Location:** Istanbul, Turkey (41.0082° N, 28.9784° E)
**Period:** 2015-01-01 to 2026-05-15 (~4,100 daily observations)

### Features used

| Category | Features |
|---|---|
| Temperature | max, min, mean, diurnal range |
| Precipitation | daily sum, 3-day avg, 7-day avg, yesterday |
| Humidity | daily mean, 7-day avg, yesterday |
| Pressure | daily mean, 7-day avg, yesterday |
| Wind | max speed, yesterday |
| Cloud cover | daily mean |
| Calendar | month, day of year, season, rainy season flag |

---

## Methods

### Train / Validation / Test split

A **time-based split** is used to prevent data leakage:

| Set | Period | Size |
|---|---|---|
| Train | 2015–2022 | ~8 years |
| Validation | 2023–2024 | ~2 years |
| Test | 2025–present | ~1.5 years |

Random shuffling is explicitly avoided — using future data to predict the past
would artificially inflate model performance.

### Models compared

| Model | Notes |
|---|---|
| Baseline (majority class) | Always predicts "no rain" — naive benchmark |
| Logistic Regression | Linear model with StandardScaler, class_weight=balanced |
| Random Forest | 200 trees, max_depth=10, class_weight=balanced |
| Gradient Boosting | 200 estimators, learning_rate=0.05, max_depth=4 |

### Evaluation metrics

**Primary metric: ROC-AUC.** Chosen over accuracy because the dataset is
imbalanced (~40% rainy days). A model that always predicts "no rain" achieves
~60% accuracy but is useless in practice.

**Supporting metrics:** F1 score, Precision, Recall, Precision-Recall curve.

### Feature engineering rationale

- **Lag features** (yesterday's values): weather is temporally autocorrelated.
- **Rolling averages** (3-day, 7-day): capture persistence of weather systems.
- **Pressure**: falling pressure is a well-known precursor to precipitation.
- **Humidity**: high atmospheric moisture precedes rainfall.
- **`is_rainy_season`**: encodes Istanbul's strong Oct–Mar wet season.
- **`temp_range`**: large diurnal range signals atmospheric instability.

---

## Limitations

- Daily granularity only — no intra-day patterns captured.
- Model is trained on Istanbul city center coordinates; microclimatic
  variation across the city (e.g. Asian vs European side) is not modeled.
- The test set (~1.5 years) is relatively short; performance estimates
  may shift with more data.
- Gradient Boosting does not natively support `class_weight`; class
  imbalance is addressed indirectly via threshold tuning in the dashboard.
- Model does not incorporate numerical weather prediction (NWP) forecasts,
  which would substantially improve skill.

---

## How to run

### Requirements

- [Docker Desktop](https://www.docker.com/products/docker-desktop/)

### Steps

```bash
git clone https://github.com/sezindemirci/istanbul-rain-prediction.git
cd istanbul-rain-prediction
docker build -t istanbul-rain-prediction .
docker run -p 8501:8501 istanbul-rain-prediction
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

The container automatically:
1. Downloads weather data from Open-Meteo API (~4,100 rows, no account needed)
2. Runs preprocessing and feature engineering
3. Trains and compares four models
4. Evaluates the best model on the held-out test set
5. Launches the Streamlit dashboard

Total startup time: ~2–3 minutes.

---

## Repository structure


---

## Author

Sezin Demirci · DS 570 Final Project · 2026
