"""
features.py — RiskRadar
-----------------------
Loads the Victorian Road Crash dataset, cleans it, engineers features,
and returns a model-ready DataFrame with a risk label target column.

Victorian Road Crash Data source:
https://opendata.transport.vic.gov.au/dataset/victoria-road-crash-data

Usage:
    from src.features import build_features, FEATURE_COLS
    df = build_features("data/victorian_road_crash_data.csv")
"""

import pandas as pd
import numpy as np


REQUIRED_COLUMNS = [
    "ACCIDENT_DATE",
    "ACCIDENT_TIME",
    "SEVERITY",
    "LIGHT_CONDITION",
    "ROAD_GEOMETRY",
    "SPEED_ZONE",
    "DEG_URBAN_NAME",
    "ACCIDENT_TYPE",
    "RUN_OFFROAD",
]

# ── Severity → risk label (string values in this dataset) ─────────────────────
SEVERITY_TO_RISK = {
    "Fatal accident":           2,
    "Serious injury accident":  2,
    "Other injury accident":    1,
    "Non injury accident":      0,
}

RISK_LABELS = {0: "low", 1: "moderate", 2: "high"}

# ── Accident type → encoded int ────────────────────────────────────────────────
ACCIDENT_TYPE_MAP = {
    "Collision with vehicle":             0,
    "Collision with a fixed object":      1,
    "Struck Pedestrian":                  2,
    "Vehicle overturned (no collision)":  3,
    "No collision and no object struck":  4,
    "Struck animal":                      5,
    "collision with some other object":   6,
    "Fall from or in moving vehicle":     7,
    "Other accident":                     8,
}

# ── High-risk accident types ───────────────────────────────────────────────────
HIGH_RISK_ACCIDENT_TYPES = {
    "Vehicle overturned (no collision)",
    "Struck Pedestrian",
    "Collision with a fixed object",
}


def load_raw(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath, low_memory=False)
    df.columns = df.columns.str.strip().str.upper()
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}\nFound: {list(df.columns)}")
    return df


def parse_datetime(df: pd.DataFrame) -> pd.DataFrame:
    # Format in this dataset: YYYY-MM-DD HH:MM:SS
    df["datetime"] = pd.to_datetime(
        df["ACCIDENT_DATE"].astype(str).str.strip() + " " +
        df["ACCIDENT_TIME"].astype(str).str.strip(),
        format="%Y-%m-%d %H:%M:%S",
        errors="coerce",
    )
    # Fallback without seconds
    mask = df["datetime"].isna()
    if mask.any():
        df.loc[mask, "datetime"] = pd.to_datetime(
            df.loc[mask, "ACCIDENT_DATE"].astype(str).str.strip() + " " +
            df.loc[mask, "ACCIDENT_TIME"].astype(str).str.strip(),
            format="%Y-%m-%d %H:%M",
            errors="coerce",
        )
    invalid = df["datetime"].isna().sum()
    if invalid > 0:
        print(f"  [warn] Dropped {invalid} rows with unparseable datetime.")
    return df.dropna(subset=["datetime"])


def engineer_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df["hour"]        = df["datetime"].dt.hour
    df["day_of_week"] = df["datetime"].dt.dayofweek  # 0=Mon, 6=Sun
    df["month"]       = df["datetime"].dt.month
    df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)

    # Melbourne peak hours: 7–9am, 4–7pm
    df["is_peak_hour"] = (
        ((df["hour"] >= 7)  & (df["hour"] <= 9)) |
        ((df["hour"] >= 16) & (df["hour"] <= 19))
    ).astype(int)

    # Night driving: 10pm–5am
    df["is_night"] = ((df["hour"] >= 22) | (df["hour"] <= 5)).astype(int)

    return df


def engineer_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    # Flat CSV has no atmospheric condition column.
    # Weather features will be populated at prediction time via Open-Meteo.
    # For training we set to 0 — the model learns from other signals.
    df["weather_risk_score"] = 0
    df["is_adverse_weather"] = 0
    return df


def engineer_road_features(df: pd.DataFrame) -> pd.DataFrame:
    # Speed zone
    df["speed_zone_num"] = pd.to_numeric(
        df["SPEED_ZONE"].astype(str).str.extract(r"(\d+)")[0],
        errors="coerce"
    ).fillna(60).astype(int)
    df["is_high_speed_zone"] = (df["speed_zone_num"] >= 80).astype(int)

    # Light condition
    dark_conditions = [
        "Dark No street lights",
        "Dark Street lights off",
        "Dark Street lights on",
        "Dark Street lights unknown",
        "Dusk/Dawn",
    ]
    df["is_dark"] = df["LIGHT_CONDITION"].astype(str).isin(dark_conditions).astype(int)

    # Road geometry: intersection flag
    intersection_types = [
        "T intersection",
        "Cross intersection",
        "Y intersection",
        "Multiple intersection",
    ]
    df["is_intersection"] = df["ROAD_GEOMETRY"].astype(str).isin(intersection_types).astype(int)

    # Heavy vehicle
    if "HEAVYVEHICLE" in df.columns:
        df["has_heavy_vehicle"] = (
            pd.to_numeric(df["HEAVYVEHICLE"], errors="coerce").fillna(0) > 0
        ).astype(int)
    else:
        df["has_heavy_vehicle"] = 0

    # Accident type — encoded int
    df["accident_type_enc"] = (
        df["ACCIDENT_TYPE"].astype(str).str.strip()
        .map(ACCIDENT_TYPE_MAP)
        .fillna(8)
        .astype(int)
    )

    # High-risk accident type flag
    df["is_high_risk_accident_type"] = (
        df["ACCIDENT_TYPE"].astype(str).str.strip()
        .isin(HIGH_RISK_ACCIDENT_TYPES)
        .astype(int)
    )

    # Run off road flag
    df["run_offroad"] = (
        df["RUN_OFFROAD"].astype(str).str.strip().str.upper() == "YES"
    ).astype(int)

    # Number of vehicles (if available)
    if "NO_OF_VEHICLES" in df.columns:
        df["no_of_vehicles"] = pd.to_numeric(
            df["NO_OF_VEHICLES"], errors="coerce"
        ).fillna(1).clip(1, 10).astype(int)
    else:
        df["no_of_vehicles"] = 1

    # Total persons involved
    if "TOTAL_PERSONS" in df.columns:
        df["total_persons"] = pd.to_numeric(
            df["TOTAL_PERSONS"], errors="coerce"
        ).fillna(1).clip(1, 20).astype(int)
    else:
        df["total_persons"] = 1

    return df


def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    df["risk_label"] = df["SEVERITY"].astype(str).str.strip().map(SEVERITY_TO_RISK)
    invalid = df["risk_label"].isna().sum()
    if invalid > 0:
        print(f"  [warn] Dropped {invalid} rows with unmapped SEVERITY values.")
    df = df.dropna(subset=["risk_label"])
    df["risk_label"] = df["risk_label"].astype(int)
    return df


def filter_melbourne(df: pd.DataFrame) -> pd.DataFrame:
    mask = df["DEG_URBAN_NAME"].astype(str).isin(["MELB_URBAN", "MELBOURNE_CBD"])
    filtered = df[mask].copy()
    print(f"  Filtered to Melbourne area: {len(filtered):,} of {len(df):,} records.")
    return filtered


def select_features(df: pd.DataFrame) -> pd.DataFrame:
    return df[FEATURE_COLS + ["risk_label"]].copy()


def build_features(
    filepath: str,
    melbourne_only: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Full pipeline: load → clean → filter → engineer → return model-ready DataFrame.
    """
    if verbose:
        print(f"Loading data from: {filepath}")
    df = load_raw(filepath)
    if verbose:
        print(f"  Raw shape: {df.shape}")

    if melbourne_only:
        df = filter_melbourne(df)

    df = parse_datetime(df)
    if verbose:
        print(f"  After datetime parse: {df.shape}")

    df = engineer_time_features(df)
    df = engineer_weather_features(df)
    df = engineer_road_features(df)
    df = encode_target(df)
    df = select_features(df)

    before = len(df)
    df = df.dropna()
    dropped = before - len(df)
    if dropped > 0 and verbose:
        print(f"  Dropped {dropped} rows with remaining NaN values.")

    if verbose:
        print(f"  Final dataset shape: {df.shape}")
        print(f"\n  Risk label distribution:")
        counts = df["risk_label"].map(RISK_LABELS).value_counts()
        for label, count in counts.items():
            pct = 100 * count / len(df)
            print(f"    {label:10s}: {count:,} ({pct:.1f}%)")

    return df


# ── Feature columns used by the model ─────────────────────────────────────────
# Keep this in sync with select_features() above.
FEATURE_COLS = [
    # Time features
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "is_peak_hour",
    "is_night",
    # Weather features (0 during training, live values at prediction time)
    "weather_risk_score",
    "is_adverse_weather",
    # Road features
    "speed_zone_num",
    "is_high_speed_zone",
    "is_dark",
    "is_intersection",
    "has_heavy_vehicle",
    # Accident context features
    "accident_type_enc",
    "is_high_risk_accident_type",
    "run_offroad",
    "no_of_vehicles",
    "total_persons",
]


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "data/victorian_road_crash_data.csv"
    df = build_features(path)
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nFeature columns:", FEATURE_COLS)
