import pandas as pd
import numpy as np

REQUIRED_COLUMNS = ["timestamp", "heart_rate", "hrv", "activity"]

def validate_data(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    validate_data(df)
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    for col in ["heart_rate", "hrv", "activity"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=REQUIRED_COLUMNS)
    out = out.sort_values("timestamp").drop_duplicates("timestamp")
    out["activity"] = out["activity"].clip(lower=0)
    return out.reset_index(drop=True)

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["hour"] = out["timestamp"].dt.hour
    out["day_of_week"] = out["timestamp"].dt.dayofweek
    out["minute_of_day"] = out["hour"] * 60 + out["timestamp"].dt.minute
    return out

def add_rolling_features(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    out = df.copy()
    min_periods = max(5, window // 4)
    out["hr_mean"] = out["heart_rate"].rolling(window, min_periods=min_periods).mean()
    out["hr_std"] = out["heart_rate"].rolling(window, min_periods=min_periods).std()
    out["hrv_mean"] = out["hrv"].rolling(window, min_periods=min_periods).mean()
    out["activity_mean"] = out["activity"].rolling(window, min_periods=min_periods).mean()
    out["hr_deviation"] = out["heart_rate"] - out["hr_mean"]
    out["hrv_deviation"] = out["hrv"] - out["hrv_mean"]
    return out
