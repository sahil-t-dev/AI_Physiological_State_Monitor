from pathlib import Path
import pandas as pd
import numpy as np


def to_ns_datetime(values):
    """Convert datetime values to consistent nanosecond precision."""
    return pd.to_datetime(values).astype("datetime64[ns]")


def get_dataset_root():
    """Locate the extracted PhysioNet dataset."""

    downloads = Path.home() / "Downloads"

    candidates = [
        downloads
        / "wearable-device-dataset-from-induced-stress-and-structured-exercise-sessions-1.0.1"
        / "wearable-device-dataset-from-induced-stress-and-structured-exercise-sessions-1.0.1"
        / "Wearable_Dataset",

        downloads
        / "wearable-device-dataset-from-induced-stress-and-structured-exercise-sessions-1.0.1"
        / "Wearable_Dataset",
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        "Wearable_Dataset was not found in your Downloads folder."
    )


def read_hr(folder):
    """Read heart-rate data from HR.csv."""

    file = folder / "HR.csv"

    with open(file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    start_time = pd.to_datetime(lines[0].strip())
    sample_rate = float(lines[1].strip())

    values = pd.to_numeric(
        pd.Series(lines[2:]),
        errors="coerce"
    ).dropna()

    timestamps = pd.date_range(
        start=start_time,
        periods=len(values),
        freq=pd.Timedelta(seconds=1 / sample_rate)
    )

    timestamps = to_ns_datetime(timestamps)

    return pd.DataFrame({
        "timestamp": timestamps,
        "heart_rate": values.to_numpy(dtype=float)
    })


def read_ibi(folder):
    """Read inter-beat intervals from IBI.csv."""

    file = folder / "IBI.csv"

    df = pd.read_csv(
        file,
        header=None,
        names=["relative_time", "ibi"]
    )

    df["relative_time"] = pd.to_numeric(
        df["relative_time"],
        errors="coerce"
    )

    df["ibi"] = pd.to_numeric(
        df["ibi"],
        errors="coerce"
    )

    return df.dropna().reset_index(drop=True)


def calculate_hrv(ibi, start_time, timestamps):
    """
    Calculate rolling RMSSD HRV from IBI intervals.

    IBI is stored in seconds.
    HRV/RMSSD is returned in milliseconds.
    """

    ibi = ibi.copy()

    start_time = pd.Timestamp(start_time)

    timestamps = to_ns_datetime(timestamps)

    # Create absolute timestamps for IBI measurements.
    ibi["timestamp"] = (
        start_time
        + pd.to_timedelta(
            ibi["relative_time"],
            unit="s"
        )
    )

    ibi["timestamp"] = to_ns_datetime(
        ibi["timestamp"]
    )

    # Remove invalid IBI values.
    ibi = ibi[
        (ibi["ibi"] >= 0.30)
        & (ibi["ibi"] <= 2.00)
    ].copy()

    # Difference between consecutive IBI values.
    ibi["diff_ms"] = (
        ibi["ibi"].diff() * 1000.0
    )

    ibi["squared_diff"] = (
        ibi["diff_ms"] ** 2
    )

    # Rolling RMSSD.
    ibi["rmssd"] = np.sqrt(
        ibi["squared_diff"]
        .rolling(
            window=60,
            min_periods=10
        )
        .mean()
    )

    # Left side: HR timestamps.
    hrv_left = pd.DataFrame({
        "timestamp": timestamps
    })

    hrv_left["timestamp"] = to_ns_datetime(
        hrv_left["timestamp"]
    )

    hrv_left = (
        hrv_left
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    # Right side: calculated HRV.
    hrv_right = (
        ibi[
            ["timestamp", "rmssd"]
        ]
        .dropna()
        .copy()
    )

    hrv_right["timestamp"] = to_ns_datetime(
        hrv_right["timestamp"]
    )

    hrv_right = (
        hrv_right
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    # Match latest HRV value to each HR timestamp.
    hrv = pd.merge_asof(
        hrv_left,
        hrv_right,
        on="timestamp",
        direction="backward"
    )

    hrv["hrv"] = hrv["rmssd"]

    # Fill beginning/end gaps.
    hrv["hrv"] = (
        hrv["hrv"]
        .bfill()
        .ffill()
    )

    return hrv[
        ["timestamp", "hrv"]
    ]


def read_acc(folder):
    """Read accelerometer data and calculate activity magnitude."""

    file = folder / "ACC.csv"

    with open(file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    start_time = pd.to_datetime(
        lines[0].split(",")[0].strip()
    )

    sample_rate = float(
        lines[1].split(",")[0].strip()
    )

    values = pd.read_csv(
        file,
        skiprows=2,
        header=None,
        names=[
            "acc_x",
            "acc_y",
            "acc_z"
        ]
    )

    values = values.apply(
        pd.to_numeric,
        errors="coerce"
    ).dropna()

    timestamps = pd.date_range(
        start=start_time,
        periods=len(values),
        freq=pd.Timedelta(
            seconds=1 / sample_rate
        )
    )

    timestamps = to_ns_datetime(timestamps)

    acc = pd.DataFrame({
        "timestamp": timestamps,
        "acc_x": values["acc_x"].to_numpy(),
        "acc_y": values["acc_y"].to_numpy(),
        "acc_z": values["acc_z"].to_numpy()
    })

    # Acceleration magnitude.
    acc["activity_raw"] = np.sqrt(
        acc["acc_x"] ** 2
        + acc["acc_y"] ** 2
        + acc["acc_z"] ** 2
    )

    # Convert accelerometer readings to 1-second activity.
    acc = (
        acc
        .set_index("timestamp")["activity_raw"]
        .resample("1s")
        .mean()
        .reset_index()
    )

    acc.rename(
        columns={
            "activity_raw": "activity"
        },
        inplace=True
    )

    acc["timestamp"] = to_ns_datetime(
        acc["timestamp"]
    )

    return acc


def load_subject(
    dataset_type="STRESS",
    subject="f01"
):
    """
    Load one PhysioNet participant/session.

    Returns:

        timestamp
        heart_rate
        hrv
        activity
    """

    root = get_dataset_root()

    folder = (
        root
        / dataset_type
        / subject
    )

    if not folder.exists():
        raise FileNotFoundError(
            f"Subject folder not found:\n{folder}"
        )

    required_files = [
        "HR.csv",
        "IBI.csv",
        "ACC.csv"
    ]

    missing = [
        filename
        for filename in required_files
        if not (folder / filename).exists()
    ]

    if missing:
        raise FileNotFoundError(
            f"Missing files in {folder}:\n{missing}"
        )

    # =========================================================
    # HEART RATE
    # =========================================================

    hr = read_hr(folder)

    # =========================================================
    # IBI -> HRV
    # =========================================================

    ibi = read_ibi(folder)

    hrv = calculate_hrv(
        ibi,
        hr["timestamp"].iloc[0],
        hr["timestamp"]
    )

    # =========================================================
    # ACC -> ACTIVITY
    # =========================================================

    acc = read_acc(folder)

    # Make absolutely sure all datetime columns match.
    hr["timestamp"] = to_ns_datetime(
        hr["timestamp"]
    )

    hrv["timestamp"] = to_ns_datetime(
        hrv["timestamp"]
    )

    acc["timestamp"] = to_ns_datetime(
        acc["timestamp"]
    )

    # =========================================================
    # HR + HRV
    # =========================================================

    result = pd.merge_asof(
        hr.sort_values("timestamp"),
        hrv.sort_values("timestamp"),
        on="timestamp",
        direction="nearest",
        tolerance=pd.Timedelta(seconds=2)
    )

    # =========================================================
    # + ACTIVITY
    # =========================================================

    result = pd.merge_asof(
        result.sort_values("timestamp"),
        acc.sort_values("timestamp"),
        on="timestamp",
        direction="nearest",
        tolerance=pd.Timedelta(seconds=2)
    )

    # =========================================================
    # FINAL DATAFRAME
    # =========================================================

    result = result[
        [
            "timestamp",
            "heart_rate",
            "hrv",
            "activity"
        ]
    ]

    result = result.dropna(
        subset=[
            "timestamp",
            "heart_rate",
            "hrv",
            "activity"
        ]
    )

    result = (
        result
        .sort_values("timestamp")
        .drop_duplicates("timestamp")
        .reset_index(drop=True)
    )

    # Ensure numeric columns.
    for column in [
        "heart_rate",
        "hrv",
        "activity"
    ]:
        result[column] = pd.to_numeric(
            result[column],
            errors="coerce"
        )

    result = result.dropna().reset_index(
        drop=True
    )

    return result


def list_subjects(dataset_type):
    """Return available participant folders."""

    root = get_dataset_root()

    folder = root / dataset_type

    if not folder.exists():
        return []

    subjects = [
        path.name
        for path in folder.iterdir()
        if (
            path.is_dir()
            and path.name.lower().startswith(
                ("f", "m")
            )
        )
    ]

    return sorted(subjects)