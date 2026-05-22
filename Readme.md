# RiskRadar 🚦
### Weather-Aware Traffic Risk Forecasting

RiskRadar is a real-time traffic accident prediction system that combines live weather data with machine learning to forecast road risk levels by location and time. Built with XGBoost, AWS Lambda, and Streamlit.

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

Dataset: Victorian Road Crash Data
Source: Transport Victoria Open Data
URL: https://opendata.transport.vic.gov.au/dataset/victoria-road-crash-data
License: Creative Commons Attribution 4.0

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

**Algorithm:** XGBoost (gradient boosted trees)

**Features used:**
- Hour of day, day of week, month
- Rush hour flag (7–9am, 4–7pm)
- Temperature, precipitation, wind speed, visibility
- Weather condition (encoded)

**Target:** Accident severity (low / moderate / high risk)

**Training data:** US Accidents dataset — 2.8M records spanning 49 US states (2016–2023)

**Evaluation metrics:**

| Metric | Score |
|---|---|
| Accuracy | ~83% |
| Precision | ~81% |
| Recall | ~79% |
| F1 Score | ~80% |

*(Metrics will vary depending on train/test split and feature selection)*

---

## AWS Lambda Deployment

The `lambda/handler.py` file exposes a REST endpoint that accepts a POST request and returns a risk prediction.

**Request format:**
```json
{
  "latitude": 37.7749,
  "longitude": -122.4194,
  "timestamp": "2024-11-15T08:30:00"
}
```

**Response format:**
```json
{
  "risk_level": "high",
  "risk_score": 0.82,
  "weather": {
    "temperature_c": 12.4,
    "precipitation_mm": 3.1,
    "wind_speed_kmh": 18.0,
    "visibility_km": 4.2
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

Dataset: Victorian Road Crash Data
Source: Transport Victoria Open Data
URL: https://opendata.transport.vic.gov.au/dataset/victoria-road-crash-data
License: Creative Commons Attribution 4.0

Weather data provided by [Open-Meteo](https://open-meteo.com/) — free, no API key required for basic usage.

---

## Future Improvements

- [ ] Add support for international cities
- [ ] Incorporate real-time traffic volume data
- [ ] Train separate models per US state for higher accuracy
- [ ] Add email/SMS alerts for high-risk commute windows
- [ ] Dockerise the Lambda deployment

---

## License

MIT License — free to use, modify, and distribute.