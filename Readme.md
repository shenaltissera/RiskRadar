# RiskRadar 🚦
### Weather-Aware Traffic Risk Forecasting

RiskRadar is a real-time traffic accident prediction system that combines live weather data with machine learning to forecast road risk levels by location and time. Built with XGBoost, AWS Lambda, and Streamlit.

<img width="3600" height="2058" alt="B2F6F296-7490-4C0D-824B-A74A6FB57241" src="https://github.com/user-attachments/assets/5fc25d15-0750-45bc-af3f-d02f9c07c4fa" />


![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![XGBoost](https://img.shields.io/badge/XGBoost-ML_Model-orange?style=flat-square)
![AWS Lambda](https://img.shields.io/badge/AWS-Lambda-yellow?style=flat-square&logo=amazonaws)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?style=flat-square&logo=streamlit)

---

## Overview

RiskRadar takes a location and timestamp, fetches live weather conditions, and returns a predicted accident risk score — low, moderate, or high. The system is served through a serverless AWS Lambda API and visualised in an interactive Streamlit dashboard.

**Key features:**
- Real-time risk predictions using live weather from the Open-Meteo API
- XGBoost classifier trained on 2.8M real US accident records
- Serverless prediction endpoint via AWS Lambda
- Interactive Streamlit dashboard with risk map and charts

---

## Architecture

```
Weather API (Open-Meteo)
        │
        ▼
Feature Engineering (Pandas)
        │
        ▼
XGBoost Model ──► joblib (saved model)
        │
        ▼
AWS Lambda (REST API endpoint)
        │
        ▼
Streamlit Dashboard (live UI)
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| ML Model | XGBoost, scikit-learn |
| Data Processing | Pandas, NumPy |
| Weather Data | Open-Meteo API |
| Training Dataset | US Accidents (Kaggle, 2.8M records) |
| Serving | AWS Lambda + API Gateway |
| Dashboard | Streamlit, PyDeck, Matplotlib |
| Model Storage | joblib |

---

## Project Structure

```
riskradar/
├── data/
│   └── us_accidents_sample.csv     # Training dataset (download separately)
├── notebooks/
│   └── exploration.ipynb           # EDA and feature analysis
├── src/
│   ├── train.py                    # Model training script
│   ├── predict.py                  # Prediction logic
│   ├── features.py                 # Feature engineering
│   └── weather.py                  # Open-Meteo API client
├── lambda/
│   ├── handler.py                  # AWS Lambda function
│   └── requirements.txt
├── dashboard/
│   └── app.py                      # Streamlit app
├── models/
│   └── riskradar_model.joblib      # Saved model (generated after training)
├── requirements.txt
└── README.md
```

---

## Getting Started

### 1. Clone and install dependencies

```bash
git clone https://github.com/yourusername/riskradar.git
cd riskradar
pip install -r requirements.txt
```

### 2. Download the training dataset

Download the [US Accidents dataset](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents) from Kaggle and place it in `data/us_accidents_sample.csv`.

### 3. Train the model

```bash
python src/train.py
```

This will preprocess the data, train the XGBoost classifier, evaluate it, and save the model to `models/riskradar_model.joblib`.

### 4. Run the Streamlit dashboard locally

```bash
streamlit run dashboard/app.py
```

---

## Model Details

**Algorithm:** XGBoost (gradient boosted trees, binary classifier)

**Features used:**
- Time — hour of day, day of week, month, peak hour flag (7–9am, 4–7pm), night flag
- Weather — risk score and adverse weather flag (live from Open-Meteo at prediction time)
- Road — speed zone, high-speed flag, dark conditions, intersection type, heavy vehicle
- Crash context — accident type, run-off-road flag, number of vehicles, persons involved

**Target:** Binary risk classification — **high** (fatal or serious injury) vs **moderate** (other injury)

**Training data:** Victorian Road Crash Data — 125,661 Melbourne-area records (2013–2023), sourced from Victoria Police reports via Transport Victoria Open Data.

**Evaluation metrics:**

| Metric | Score |
|---|---|
| Accuracy | 61.5% |
| ROC-AUC | 0.617 |
| CV AUC (5-fold) | 0.614 ± 0.001 |
| High risk precision | 0.45 |
| High risk recall | 0.51 |

**A note on model performance:** The Victorian crash dataset records *what happened* — accident type, time, road geometry — but crash severity is primarily driven by factors not captured here: vehicle speed at impact, seatbelt use, and driver state. The strongest available feature (accident type) correlates at 0.14 with severity. The model's value is therefore in **real-time contextual risk scoring** — combining live weather, time of day, and road conditions — rather than precise severity prediction. See Future Improvements for a planned approach that better addresses this.

---

## AWS Lambda Deployment

The `lambda/handler.py` file exposes a REST endpoint that accepts a POST request and returns a risk prediction.

**Request format:**
```json
{
  "latitude": -37.8136,
  "longitude": 144.9631,
  "timestamp": "2024-11-15T08:30:00"
}
```

**Response format:**
```json
{
  "risk_level": "high",
  "risk_score": 0.74,
  "weather": {
    "temperature_c": 12.4,
    "precipitation_mm": 3.1,
    "wind_speed_kmh": 18.0,
    "weather_condition": "Rain"
  }
}
```

**Deploy to Lambda:**
```bash
cd lambda
pip install -r requirements.txt -t ./package
zip -r function.zip handler.py riskradar_model.joblib package/
aws lambda update-function-code --function-name riskradar --zip-file fileb://function.zip
```

---

## Dashboard

The Streamlit dashboard (`dashboard/app.py`) includes:

- **Risk map** — colour-coded map showing predicted risk by area
- **Live prediction panel** — enter a location and time to get an instant risk score
- **Risk by hour chart** — visualise how accident risk changes through the day
- **Weather overlay** — current conditions for the selected location

---

## Requirements

```
xgboost>=1.7
pandas>=2.0
scikit-learn>=1.3
streamlit>=1.28
requests>=2.31
joblib>=1.3
matplotlib>=3.7
pydeck>=0.8
boto3>=1.28
numpy>=1.24
```

---

## Dataset Credit

Victorian Road Crash Data, Transport Victoria Open Data Portal.  
[https://opendata.transport.vic.gov.au/dataset/victoria-road-crash-data](https://opendata.transport.vic.gov.au/dataset/victoria-road-crash-data)  
License: Creative Commons Attribution 4.0. Data consolidated from Victoria Police reports and hospital injury records, updated monthly.

Weather data provided by [Open-Meteo](https://open-meteo.com/) — free, open-source, no API key required.

---

## Future Improvements

- [ ] **Accident likelihood model** — rather than predicting severity after a crash, predict the *probability of an accident occurring* at a given location, time, and weather condition by generating negative samples (no-crash observations). This approach is better suited to the available data and expected to achieve ROC-AUC 0.75–0.85.
- [ ] Incorporate real-time traffic volume data (VicRoads API)
- [ ] Add pedestrian and cyclist risk as separate model outputs
- [ ] Add email/SMS alerts for high-risk commute windows
- [ ] Expand coverage to all of Victoria, not just Melbourne
- [ ] Dockerise the Lambda deployment

---

## License

MIT License — free to use, modify, and distribute.
