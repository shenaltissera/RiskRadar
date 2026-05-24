"""
handler.py — RiskRadar AWS Lambda
----------------------------------
Serverless REST API endpoint for traffic accident risk prediction.
Deployed via AWS Lambda + API Gateway.

Accepts POST requests with location and optional parameters,
returns a risk prediction using the trained XGBoost model.

Request body (JSON):
    {
        "latitude":          -37.8136,   # required
        "longitude":         144.9631,   # required
        "datetime":          "2024-11-15T08:30:00",  # optional, default: now
        "speed_zone":        60,         # optional, default: 60
        "is_intersection":   false,      # optional, default: false
        "has_heavy_vehicle": false       # optional, default: false
    }

Response (JSON):
    {
        "risk_level":  "high",
        "risk_score":  0.74,
        "weather": {
            "temperature_c":     12.4,
            "precipitation_mm":  3.1,
            "wind_speed_kmh":    18.0,
            "weather_condition": "Rain",
            "weather_risk_score": 2,
            "is_adverse_weather": 1
        },
        "timestamp":     "2024-11-15T08:30:00",
        "request_id":    "abc-123",
        "model_version": "2024-11-10T14:22:31"
    }

Deploy:
    cd lambda
    pip install -r requirements.txt -t ./package
    zip -r function.zip handler.py riskradar_model.joblib package/
    aws lambda update-function-code \
        --function-name riskradar \
        --zip-file fileb://function.zip
"""

import json
import os
import sys
import traceback
import uuid
from datetime import datetime

# Lambda packages are zipped into ./package — add to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "package"))
sys.path.insert(0, os.path.dirname(__file__))

# Add project root so src/ is importable both locally and in Lambda
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.predict import predict_risk

# Model is packaged alongside handler.py in the zip
MODEL_PATH = os.path.join(os.path.dirname(__file__), "riskradar_model.joblib")

# Melbourne bounding box — reject obviously wrong coordinates
LAT_MIN, LAT_MAX = -39.5, -36.0
LON_MIN, LON_MAX = 143.5, 147.5


# ── CORS headers ───────────────────────────────────────────────────────────────
CORS_HEADERS = {
    "Content-Type":                "application/json",
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type",
}


def _response(status_code: int, body: dict) -> dict:
    return {
        "statusCode": status_code,
        "headers":    CORS_HEADERS,
        "body":       json.dumps(body),
    }


def _error(status_code: int, message: str, request_id: str = None) -> dict:
    body = {"error": message}
    if request_id:
        body["request_id"] = request_id
    return _response(status_code, body)


def _validate(body: dict) -> str | None:
    """Validate request body. Returns error message string or None if valid."""
    if "latitude" not in body or "longitude" not in body:
        return "Missing required fields: 'latitude' and 'longitude'"

    try:
        lat = float(body["latitude"])
        lon = float(body["longitude"])
    except (TypeError, ValueError):
        return "'latitude' and 'longitude' must be numeric"

    if not (LAT_MIN <= lat <= LAT_MAX):
        return f"'latitude' must be within Victoria ({LAT_MIN} to {LAT_MAX})"

    if not (LON_MIN <= lon <= LON_MAX):
        return f"'longitude' must be within Victoria ({LON_MIN} to {LON_MAX})"

    if "speed_zone" in body:
        try:
            speed = int(body["speed_zone"])
            if speed not in [30, 40, 50, 60, 70, 80, 90, 100, 110]:
                return "'speed_zone' must be a standard speed limit (30–110 km/h)"
        except (TypeError, ValueError):
            return "'speed_zone' must be an integer"

    if "datetime" in body:
        try:
            datetime.fromisoformat(str(body["datetime"]))
        except ValueError:
            return "'datetime' must be ISO format: YYYY-MM-DDTHH:MM:SS"

    return None


def lambda_handler(event: dict, context) -> dict:
    """
    AWS Lambda entry point.

    Handles:
        OPTIONS  — CORS preflight
        POST     — risk prediction
        anything else — 405 Method Not Allowed
    """
    request_id = str(uuid.uuid4())[:8]

    # ── CORS preflight ─────────────────────────────────────────────────────────
    http_method = event.get("httpMethod", "POST")
    if http_method == "OPTIONS":
        return _response(200, {"message": "OK"})

    if http_method != "POST":
        return _error(405, f"Method '{http_method}' not allowed. Use POST.", request_id)

    # ── Parse body ─────────────────────────────────────────────────────────────
    try:
        raw_body = event.get("body", "{}")
        if isinstance(raw_body, str):
            body = json.loads(raw_body)
        else:
            body = raw_body or {}
    except json.JSONDecodeError:
        return _error(400, "Request body must be valid JSON.", request_id)

    # ── Validate ───────────────────────────────────────────────────────────────
    validation_error = _validate(body)
    if validation_error:
        return _error(400, validation_error, request_id)

    # ── Extract parameters ─────────────────────────────────────────────────────
    lat               = float(body["latitude"])
    lon               = float(body["longitude"])
    speed_zone        = int(body.get("speed_zone", 60))
    is_intersection   = bool(body.get("is_intersection", False))
    has_heavy_vehicle = bool(body.get("has_heavy_vehicle", False))

    dt = None
    if "datetime" in body:
        dt = datetime.fromisoformat(str(body["datetime"]))

    # ── Run prediction ─────────────────────────────────────────────────────────
    try:
        result = predict_risk(
            lat=lat,
            lon=lon,
            dt=dt,
            speed_zone=speed_zone,
            is_intersection=is_intersection,
            has_heavy_vehicle=has_heavy_vehicle,
            model_path=MODEL_PATH,
        )
    except FileNotFoundError as e:
        return _error(503, f"Model unavailable: {str(e)}", request_id)
    except Exception as e:
        print(f"[ERROR] request_id={request_id} | {traceback.format_exc()}")
        return _error(500, "Internal prediction error. Please try again.", request_id)

    # ── Build response ─────────────────────────────────────────────────────────
    weather = result["weather"]

    response_body = {
        "risk_level":  result["risk_level"],
        "risk_score":  result["risk_score"],
        "weather": {
            "temperature_c":      weather.get("temperature_c"),
            "precipitation_mm":   weather.get("precipitation_mm"),
            "wind_speed_kmh":     weather.get("wind_speed_kmh"),
            "weather_condition":  weather.get("weather_condition"),
            "weather_risk_score": weather.get("weather_risk_score"),
            "is_adverse_weather": weather.get("is_adverse_weather"),
        },
        "timestamp":     result["timestamp"],
        "request_id":    request_id,
        "model_version": result.get("model_version", "unknown"),
    }

    return _response(200, response_body)


# ── Local test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing Lambda handler locally...\n")

    test_events = [
        {
            "name": "Melbourne CBD — current time",
            "event": {
                "httpMethod": "POST",
                "body": json.dumps({
                    "latitude":  -37.8136,
                    "longitude": 144.9631,
                    "speed_zone": 60,
                }),
            },
        },
        {
            "name": "Tullamarine Freeway — night, high speed",
            "event": {
                "httpMethod": "POST",
                "body": json.dumps({
                    "latitude":   -37.7100,
                    "longitude":  144.8800,
                    "speed_zone": 100,
                    "datetime":   "2024-11-15T02:30:00",
                }),
            },
        },
        {
            "name": "Missing coordinates — should return 400",
            "event": {
                "httpMethod": "POST",
                "body": json.dumps({"speed_zone": 60}),
            },
        },
        {
            "name": "CORS preflight",
            "event": {"httpMethod": "OPTIONS"},
        },
    ]

    for test in test_events:
        print(f"Test: {test['name']}")
        result = lambda_handler(test["event"], context=None)
        body = json.loads(result["body"])
        print(f"  Status : {result['statusCode']}")
        if "risk_level" in body:
            print(f"  Risk   : {body['risk_level']} (score: {body['risk_score']})")
            print(f"  Weather: {body['weather']['weather_condition']}")
        else:
            print(f"  Result : {body}")
        print()