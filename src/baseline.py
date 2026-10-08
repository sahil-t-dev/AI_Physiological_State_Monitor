import pandas as pd
import numpy as np

def create_personal_baseline(df: pd.DataFrame, warmup: int = 20) -> dict:
    """Create a simple baseline from the first warmup observations."""
    data = df.head(max(warmup, 1))
    return {
        "heart_rate_mean": float(data["heart_rate"].mean()),
        "heart_rate_std": float(data["heart_rate"].std(ddof=0) or 1.0),
        "hrv_mean": float(data["hrv"].mean()),
        "hrv_std": float(data["hrv"].std(ddof=0) or 1.0),
        "activity_mean": float(data["activity"].mean()),
        "activity_std": float(data["activity"].std(ddof=0) or 1.0),
    }

def add_baseline_deviation(df: pd.DataFrame, baseline: dict) -> pd.DataFrame:
    out = df.copy()
    out["baseline_hr_z"] = (
        (out["heart_rate"] - baseline["heart_rate_mean"])
        / max(baseline["heart_rate_std"], 1e-6)
    )
    out["baseline_hrv_z"] = (
        (out["hrv"] - baseline["hrv_mean"])
        / max(baseline["hrv_std"], 1e-6)
    )
    out["baseline_activity_z"] = (
        (out["activity"] - baseline["activity_mean"])
        / max(baseline["activity_std"], 1e-6)
    )
    return out
