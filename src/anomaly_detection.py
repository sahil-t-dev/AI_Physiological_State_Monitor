import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

FEATURES = [
    "heart_rate",
    "hrv",
    "activity",
    "hr_deviation",
    "hrv_deviation",
    "baseline_hr_z",
    "baseline_hrv_z",
    "baseline_activity_z",
]

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in FEATURES:
        if c not in out:
            out[c] = 0.0
    return out[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0)

def fit_isolation_forest(df: pd.DataFrame, contamination: float = 0.05, random_state: int = 42):
    X = prepare_features(df)
    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X)
    pred = model.predict(X)
    score = -model.decision_function(X)
    result = df.copy()
    result["anomaly_score"] = score
    result["anomaly"] = pred == -1
    return model, result

def predict(model, df: pd.DataFrame) -> pd.DataFrame:
    X = prepare_features(df)
    result = df.copy()
    result["anomaly_score"] = -model.decision_function(X)
    result["anomaly"] = model.predict(X) == -1
    return result
