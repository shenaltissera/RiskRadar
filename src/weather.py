"""
weather.py — RiskRadar
----------------------
Fetches live and historical weather for Melbourne from the Open-Meteo API.
No API key required — Open-Meteo is free and open source.

API docs: https://open-meteo.com/en/docs

Usage:
    from src.weather import get_current_weather, weather_to_features

    weather = get_current_weather()
    features = weather_to_features(weather)
"""

import requests
from datetime import datetime, timezone
from typing import Optional

# ── Melbourne defaults ─────────────────────────────────────────────────────────
MELBOURNE_LAT = -37.8136
MELBOURNE_LON = 144.9631

# Open-Meteo base URLs
FORECAST_URL  = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL   = "https://archive-api.open-meteo.com/v1/archive"

# Request timeout (seconds)
TIMEOUT = 10


# ── Weather condition → risk score mapping ─────────────────────────────────────
# WMO weather interpretation codes from Open-Meteo:
# https://open-meteo.com/en/docs#weathervariables

WMO_TO_CONDITION = {
    0:  "Clear",
    1:  "Mainly clear",
    2:  "Partly cloudy",
    3:  "Overcast",
    45: "Fog",
    48: "Icy fog",
    51: "Light drizzle",
    53: "Drizzle",
    55: "Heavy drizzle",
    61: "Light rain",
    63: "Rain",
    65: "Heavy rain",
    71: "Light snow",
    73: "Snow",
    75: "Heavy snow",
    77: "Snow grains",
    80: "Light showers",
    81: "Showers",
    82: "Heavy showers",
    85: "Snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with hail",
    99: "Thunderstorm with heavy hail",
}

# Risk score: 0 = clear, 1 = minor risk, 2 = moderate risk, 3 = high risk
WMO_TO_RISK = {
    0:  0,   # Clear
    1:  0,   # Mainly clear
    2:  0,   # Partly cloudy
    3:  0,   # Overcast
    45: 3,   # Fog
    48: 3,   # Icy fog
    51: 1,   # Light drizzle
    53: 2,   # Drizzle
    55: 2,   # Heavy drizzle
    61: 2,   # Light rain
    63: 2,   # Rain
    65: 3,   # Heavy rain
    71: 2,   # Light snow
    73: 3,   # Snow
    75: 3,   # Heavy snow
    77: 2,   # Snow grains
    80: 1,   # Light showers
    81: 2,   # Showers
    82: 3,   # Heavy showers
    85: 2,   # Snow showers
    86: 3,   # Heavy snow showers
    95: 3,   # Thunderstorm
    96: 3,   # Thunderstorm with hail
    99: 3,   # Thunderstorm with heavy hail
}


def get_current_weather(
    lat: float = MELBOURNE_LAT,
    lon: float = MELBOURNE_LON,
) -> dict:
    """
    Fetch current weather conditions from Open-Meteo for a given location.

    Args:
        lat: Latitude (default: Melbourne CBD)
        lon: Longitude (default: Melbourne CBD)

    Returns:
        dict with weather fields, or fallback defaults on API failure

    Example return:
        {
            "temperature_c": 14.2,
            "precipitation_mm": 0.0,
            "wind_speed_kmh": 18.5,
            "visibility_m": 10000,
            "weather_code": 3,
            "weather_condition": "Overcast",
            "weather_risk_score": 0,
            "is_adverse_weather": 0,
            "timestamp": "2024-11-15T08:30:00+00:00",
        }
    """
    params = {
        "latitude":           lat,
        "longitude":          lon,
        "current":            [
            "temperature_2m",
            "precipitation",
            "wind_speed_10m",
            "visibility",
            "weather_code",
        ],
        "wind_speed_unit":    "kmh",
        "timezone":           "Australia/Melbourne",
    }

    try:
        response = requests.get(FORECAST_URL, params=params, timeout=TIMEOUT)
        response.raise_for_status()
        data = response.json()

        current = data.get("current", {})

        weather_code      = int(current.get("weather_code", 0))
        temperature_c     = float(current.get("temperature_2m", 15.0))
        precipitation_mm  = float(current.get("precipitation", 0.0))
        wind_speed_kmh    = float(current.get("wind_speed_10m", 10.0))
        visibility_m      = float(current.get("visibility", 10000.0))
        timestamp         = current.get("time", datetime.now(timezone.utc).isoformat())

        weather_condition    = WMO_TO_CONDITION.get(weather_code, "Unknown")
        weather_risk_score   = WMO_TO_RISK.get(weather_code, 0)
        is_adverse_weather   = int(weather_risk_score >= 2)

        return {
            "temperature_c":       temperature_c,
            "precipitation_mm":    precipitation_mm,
            "wind_speed_kmh":      wind_speed_kmh,
            "visibility_m":        visibility_m,
            "weather_code":        weather_code,
            "weather_condition":   weather_condition,
            "weather_risk_score":  weather_risk_score,
            "is_adverse_weather":  is_adverse_weather,
            "timestamp":           timestamp,
            "source":              "open-meteo",
        }

    except requests.exceptions.Timeout:
        print("[weather] API timeout — using fallback defaults.")
        return _fallback_weather()

    except requests.exceptions.RequestException as e:
        print(f"[weather] API error: {e} — using fallback defaults.")
        return _fallback_weather()

    except (KeyError, ValueError, TypeError) as e:
        print(f"[weather] Parse error: {e} — using fallback defaults.")
        return _fallback_weather()


def get_weather_for_datetime(
    dt: datetime,
    lat: float = MELBOURNE_LAT,
    lon: float = MELBOURNE_LON,
) -> dict:
    """
    Fetch historical weather for a past datetime using the Open-Meteo archive API.
    Use this when predicting risk for a specific past date/time.

    Args:
        dt:  A datetime object (aware or naive — treated as Melbourne local time)
        lat: Latitude
        lon: Longitude

    Returns:
        Same dict structure as get_current_weather()
    """
    date_str = dt.strftime("%Y-%m-%d")

    params = {
        "latitude":        lat,
        "longitude":       lon,
        "start_date":      date_str,
        "end_date":        date_str,
        "hourly":          [
            "temperature_2m",
            "precipitation",
            "wind_speed_10m",
            "visibility",
            "weather_code",
        ],
        "wind_speed_unit": "kmh",
        "timezone":        "Australia/Melbourne",
    }

    try:
        response = requests.get(ARCHIVE_URL, params=params, timeout=TIMEOUT)
        response.raise_for_status()
        data = response.json()

        hourly = data.get("hourly", {})
        times  = hourly.get("time", [])

        # Find the closest hour to the requested datetime
        target_hour = dt.strftime("%Y-%m-%dT%H:00")
        idx = 0
        if target_hour in times:
            idx = times.index(target_hour)

        def _get(key, default):
            vals = hourly.get(key, [])
            return vals[idx] if idx < len(vals) and vals[idx] is not None else default

        weather_code      = int(_get("weather_code", 0))
        temperature_c     = float(_get("temperature_2m", 15.0))
        precipitation_mm  = float(_get("precipitation", 0.0))
        wind_speed_kmh    = float(_get("wind_speed_10m", 10.0))
        visibility_m      = float(_get("visibility", 10000.0))

        weather_condition  = WMO_TO_CONDITION.get(weather_code, "Unknown")
        weather_risk_score = WMO_TO_RISK.get(weather_code, 0)
        is_adverse_weather = int(weather_risk_score >= 2)

        return {
            "temperature_c":       temperature_c,
            "precipitation_mm":    precipitation_mm,
            "wind_speed_kmh":      wind_speed_kmh,
            "visibility_m":        visibility_m,
            "weather_code":        weather_code,
            "weather_condition":   weather_condition,
            "weather_risk_score":  weather_risk_score,
            "is_adverse_weather":  is_adverse_weather,
            "timestamp":           times[idx] if idx < len(times) else date_str,
            "source":              "open-meteo-archive",
        }

    except requests.exceptions.RequestException as e:
        print(f"[weather] Archive API error: {e} — using fallback defaults.")
        return _fallback_weather()

    except (KeyError, ValueError, TypeError, IndexError) as e:
        print(f"[weather] Archive parse error: {e} — using fallback defaults.")
        return _fallback_weather()


def weather_to_features(weather: dict) -> dict:
    """
    Convert a weather dict (from get_current_weather or get_weather_for_datetime)
    into the feature dict expected by the RiskRadar model.

    This is what predict.py will call to merge live weather into the feature vector.

    Args:
        weather: dict returned by get_current_weather()

    Returns:
        dict with keys: weather_risk_score, is_adverse_weather
    """
    return {
        "weather_risk_score":  weather.get("weather_risk_score", 0),
        "is_adverse_weather":  weather.get("is_adverse_weather", 0),
    }


def _fallback_weather() -> dict:
    """Return safe default weather values when the API is unavailable."""
    return {
        "temperature_c":       15.0,
        "precipitation_mm":    0.0,
        "wind_speed_kmh":      10.0,
        "visibility_m":        10000.0,
        "weather_code":        0,
        "weather_condition":   "Clear",
        "weather_risk_score":  0,
        "is_adverse_weather":  0,
        "timestamp":           datetime.now(timezone.utc).isoformat(),
        "source":              "fallback",
    }


def format_weather_summary(weather: dict) -> str:
    """Return a human-readable weather summary for the Streamlit dashboard."""
    return (
        f"{weather['weather_condition']} · "
        f"{weather['temperature_c']:.1f}°C · "
        f"Wind {weather['wind_speed_kmh']:.0f} km/h · "
        f"Rain {weather['precipitation_mm']:.1f} mm"
    )


# ── Quick test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Fetching current Melbourne weather...\n")
    weather = get_current_weather()

    print("Current weather:")
    for k, v in weather.items():
        print(f"  {k:22s}: {v}")

    print()
    print("Model features:")
    features = weather_to_features(weather)
    for k, v in features.items():
        print(f"  {k:22s}: {v}")

    print()
    print("Summary:", format_weather_summary(weather))