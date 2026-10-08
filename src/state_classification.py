import pandas as pd
import numpy as np


def classify_state(row):
    """
    Classify the current physiological pattern using:
    - Heart-rate deviation from personal baseline
    - HRV deviation from personal baseline
    - Activity level
    - Isolation Forest anomaly result

    This is an analytical classification for the project,
    not a medical diagnosis.
    """

    hr_z = float(row.get("baseline_hr_z", 0.0))
    hrv_z = float(row.get("baseline_hrv_z", 0.0))
    activity_z = float(row.get("baseline_activity_z", 0.0))
    is_anomaly = bool(row.get("anomaly", False))

    # ---------------------------------------------------------
    # 1. Significant multivariate deviation
    # ---------------------------------------------------------
    if is_anomaly and abs(hr_z) >= 4:
        return "Significant Deviation"

    if is_anomaly and abs(hrv_z) >= 4:
        return "Significant Deviation"

    # ---------------------------------------------------------
    # 2. High physical activity
    # ---------------------------------------------------------
    if activity_z >= 2.0 and hr_z >= 1.0:
        return "High Activity"

    # ---------------------------------------------------------
    # 3. Elevated physiological activity pattern
    # ---------------------------------------------------------
    if hr_z >= 2.0 and hrv_z <= -1.0:
        return "Elevated Physiological Activity"

    if hr_z >= 2.5:
        return "Elevated Physiological Activity"

    # ---------------------------------------------------------
    # 4. Recovery / relaxed pattern
    # ---------------------------------------------------------
    if hrv_z >= 1.5 and hr_z <= 0.5 and activity_z <= 1.0:
        return "Recovery / Relaxed"

    # ---------------------------------------------------------
    # 5. General deviation
    # ---------------------------------------------------------
    if is_anomaly:
        return "Deviation Detected"

    # ---------------------------------------------------------
    # 6. Normal
    # ---------------------------------------------------------
    return "Normal"


def add_state_classification(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add physiological state classification to the dataframe.
    """

    out = df.copy()

    out["physiological_state"] = out.apply(
        classify_state,
        axis=1
    )

    return out


def state_summary(df: pd.DataFrame) -> pd.Series:
    """
    Return the number of observations in each physiological state.
    """

    if "physiological_state" not in df.columns:
        return pd.Series(dtype=int)

    return (
        df["physiological_state"]
        .value_counts()
    )