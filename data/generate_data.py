from pathlib import Path
import numpy as np
import pandas as pd

def generate(n=1500, seed=42):
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range("2026-01-01 08:00:00", periods=n, freq="5min")
    t = np.arange(n)

    # Smooth personal pattern plus daily activity variation.
    daily = 5 * np.sin(2 * np.pi * (t % 288) / 288)
    activity = np.clip(
        0.25
        + 0.18 * np.sin(2 * np.pi * (t % 288) / 288 - 0.8)
        + rng.normal(0, 0.08, n),
        0,
        1,
    )
    hr = 72 + daily + 16 * activity + rng.normal(0, 2.2, n)
    hrv = 52 - 10 * activity + rng.normal(0, 4, n)

    # Inject a few unusual combinations for anomaly-detection demonstration.
    anomaly_indices = [350, 351, 352, 900, 901, 902, 1200]
    for i in anomaly_indices:
        if i < n:
            hr[i] += 35
            hrv[i] -= 22
            activity[i] = max(0, activity[i] * 0.25)

    df = pd.DataFrame({
        "timestamp": timestamps,
        "heart_rate": np.round(hr, 2),
        "hrv": np.round(np.clip(hrv, 8, None), 2),
        "activity": np.round(activity, 3),
    })
    return df

if __name__ == "__main__":
    out = Path(__file__).resolve().parent / "sample_physiological_data.csv"
    generate().to_csv(out, index=False)
    print(f"Saved {out}")
