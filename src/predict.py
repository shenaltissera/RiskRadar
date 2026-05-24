"""
predict.py — RiskRadar
----------------------
Loads the trained XGBoost model and runs predictions.
Combines time-based features with live Melbourne weather from Open-Meteo.

Usage (from project root):
    python src/predict.py
    python src/predict.py --lat -37.8136 --lon 144.9631

In code:
    from src.predict import predict_risk
    result = predict_risk(lat=-37.8136, lon=144.9631)
"""

import argparse
import os
import sys
from datetime import datetime

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.weather import get_current_weather, get_weather_for_datetime, weather_to_features, format_weather_summary

DEFAULT_MODEL  = "models/riskradar_model.joblib"
MELBOURNE_LAT  = -37.8136
MELBOURNE_LON  = 144.9631


# ── Low-risk rule ──────────────────────────────────────────────────────────────
# The model is binary (high vs moderate). We assign "low" risk via a
# simple rule when all conditions are clearly safe — no model needed.

def _is_low_risk(hour: int, speed_zone: int, weather: dict) -> bool:
    """Return True if conditions are clearly low-risk (daytime, slow, clear)."""
    is_daytime      = 6 <= hour <= 20
    is_slow_zone    = speed_zone <= 50
    is_clear        = weather.get("weather_risk_score", 0) == 0
    is_dry          = weather.get("precipitation_mm", 0.0) < 0.5
    return is_daytime and is_slow_zone and is_clear and is_dry


# ── Feature builder for a single prediction ───────────────────────────────────

def _build_prediction_features(
    dt: datetime,
    weather: dict,
    speed_zone: int = 60,
    is_intersection: bool = False,
    has_heavy_vehicle: bool = False,
    accident_type_enc: int = 0,
    is_high_risk_accident_type: bool = False,
    run_offroad: bool = False,
    no_of_vehicles: int = 1,
    total_persons: int = 1,
) -> pd.DataFrame:
    """
    Build a single-row DataFrame of model features for one prediction.
    Time features come from dt; weather features come from the weather dict.
    Road/context features are passed in directly (used by the Streamlit UI).
    """
    hour        = dt.hour
    day_of_week = dt.weekday()       # 0=Mon, 6=Sun
    month       = dt.month
    is_weekend  = int(day_of_week >= 5)
    is_peak     = int((7 <= hour <= 9) or (16 <= hour <= 19))
    is_night    = int(hour >= 22 or hour <= 5)
    is_dark     = int(hour >= 20 or hour <= 6)

    weather_feats = weather_to_features(weather)

    row = {
        # Time
        "hour":                         hour,
        "day_of_week":                  day_of_week,
        "month":                        month,
        "is_weekend":                   is_weekend,
        "is_peak_hour":                 is_peak,
        "is_night":                     is_night,
        # Weather (live from Open-Meteo)
        "weather_risk_score":           weather_feats["weather_risk_score"],
        "is_adverse_weather":           weather_feats["is_adverse_weather"],
        # Road
        "speed_zone_num":               int(speed_zone),
        "is_high_speed_zone":           int(speed_zone >= 80),
        "is_dark":                      is_dark,
        "is_intersection":              int(is_intersection),
        "has_heavy_vehicle":            int(has_heavy_vehicle),
        # Crash context
        "accident_type_enc":            int(accident_type_enc),
        "is_high_risk_accident_type":   int(is_high_risk_accident_type),
        "run_offroad":                  int(run_offroad),
        "no_of_vehicles":               int(np.clip(no_of_vehicles, 1, 10)),
        "total_persons":                int(np.clip(total_persons, 1, 20)),
    }

    return pd.DataFrame([row])


# ── Main prediction function ───────────────────────────────────────────────────

def predict_risk(
    lat: float = MELBOURNE_LAT,
    lon: float = MELBOURNE_LON,
    dt: datetime = None,
    speed_zone: int = 60,
    is_intersection: bool = False,
    has_heavy_vehicle: bool = False,
    model_path: str = DEFAULT_MODEL,
) -> dict:
    """
    Predict traffic accident risk for a Melbourne location and time.

    Args:
        lat:              Latitude (default: Melbourne CBD)
        lon:              Longitude (default: Melbourne CBD)
        dt:               Datetime to predict for (default: now)
        speed_zone:       Speed limit in km/h (default: 60)
        is_intersection:  Is the location an intersection? (default: False)
        has_heavy_vehicle: Are heavy vehicles present? (default: False)
        model_path:       Path to saved model joblib file

    Returns:
        dict with keys:
            risk_level    — "low", "moderate", or "high"
            risk_score    — float 0.0–1.0 (probability of high risk)
            weather       — dict of current weather conditions
            features      — dict of features used for this prediction
            timestamp     — ISO string of the prediction datetime
            model_version — training timestamp from saved model
    """

    # ── Load model ─────────────────────────────────────────────────────────────
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at '{model_path}'. "
            f"Run 'python src/train.py' first."
        )

    payload      = joblib.load(model_path)
    model        = payload["model"]
    feature_cols = payload["feature_cols"]
    trained_on   = payload.get("trained_on", "unknown")

    # ── Time ───────────────────────────────────────────────────────────────────
    if dt is None:
        dt = datetime.now()

    # ── Fetch live weather ─────────────────────────────────────────────────────
    now = datetime.now()
    delta_hours = (dt - now).total_seconds() / 3600  # positive = future, negative = past

    if delta_hours < -6:
        # More than 6 hours in the past — use archive API
        weather = get_weather_for_datetime(dt, lat=lat, lon=lon)
    else:
        # Current time, near past (<6h), or any future time — use forecast API
        # Open-Meteo forecast covers up to 16 days ahead
        weather = get_current_weather(lat=lat, lon=lon)

    # ── Low-risk rule ──────────────────────────────────────────────────────────
    if _is_low_risk(dt.hour, speed_zone, weather):
        return {
            "risk_level":    "low",
            "risk_score":    0.10,
            "weather":       weather,
            "features":      {"note": "low-risk rule applied — no model call needed"},
            "timestamp":     dt.isoformat(),
            "model_version": trained_on,
        }

    # ── Build features ─────────────────────────────────────────────────────────
    X = _build_prediction_features(
        dt=dt,
        weather=weather,
        speed_zone=speed_zone,
        is_intersection=is_intersection,
        has_heavy_vehicle=has_heavy_vehicle,
    )

    # Ensure column order matches training
    X = X[feature_cols]

    # ── Predict ────────────────────────────────────────────────────────────────
    risk_score = float(model.predict_proba(X)[0][1])  # P(high risk)

    # Threshold: above 0.45 → high, else moderate
    # (lowered from 0.5 to improve recall for high-risk cases)
    risk_level = "high" if risk_score >= 0.45 else "moderate"

    return {
        "risk_level":    risk_level,
        "risk_score":    round(risk_score, 4),
        "weather":       weather,
        "features":      X.iloc[0].to_dict(),
        "timestamp":     dt.isoformat(),
        "model_version": trained_on,
    }


def format_result(result: dict) -> str:
    """Pretty-print a prediction result for CLI output."""
    risk_icons = {"low": "🟢", "moderate": "🟡", "high": "🔴"}
    icon = risk_icons.get(result["risk_level"], "⚪")

    lines = [
        "",
        "─" * 45,
        f"  RiskRadar Prediction",
        "─" * 45,
        f"  Risk level  : {icon} {result['risk_level'].upper()}",
        f"  Risk score  : {result['risk_score']:.2f}  (0=safe, 1=high risk)",
        f"  Time        : {result['timestamp']}",
        "─" * 45,
        f"  Weather     : {format_weather_summary(result['weather'])}",
        f"  Source      : {result['weather'].get('source', 'unknown')}",
        "─" * 45,
    ]
    return "\n".join(lines)


# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RiskRadar — predict traffic accident risk")
    parser.add_argument("--lat",   type=float, default=MELBOURNE_LAT, help="Latitude")
    parser.add_argument("--lon",   type=float, default=MELBOURNE_LON, help="Longitude")
    parser.add_argument("--speed", type=int,   default=60,            help="Speed zone (km/h)")
    parser.add_argument("--model", type=str,   default=DEFAULT_MODEL, help="Model path")
    parser.add_argument(
        "--datetime",
        type=str,
        default=None,
        help="Datetime string e.g. '2024-11-15 08:30' (default: now)"
    )
    args = parser.parse_args()

    dt = None
    if args.datetime:
        dt = datetime.strptime(args.datetime, "%Y-%m-%d %H:%M")

    print(f"\nFetching weather for ({args.lat}, {args.lon})...")
    result = predict_risk(
        lat=args.lat,
        lon=args.lon,
        dt=dt,
        speed_zone=args.speed,
        model_path=args.model,
    )

    print(format_result(result))