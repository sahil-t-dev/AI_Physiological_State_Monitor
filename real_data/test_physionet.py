from pathlib import Path
import pandas as pd
import numpy as np

# Your downloaded dataset
DATASET = Path.home() / "Downloads" / \
    "wearable-device-dataset-from-induced-stress-and-structured-exercise-sessions-1.0.1" / \
    "wearable-device-dataset-from-induced-stress-and-structured-exercise-sessions-1.0.1" / \
    "Wearable_Dataset"

STRESS = DATASET / "STRESS"


def read_hr(folder):
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

    return pd.DataFrame({
        "timestamp": timestamps,
        "heart_rate": values.to_numpy()
    })


def read_ibi(folder):
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

    return df.dropna()


def read_acc(folder):
    file = folder / "ACC.csv"

    with open(file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    start_time = pd.to_datetime(lines[0].split(",")[0])
    sample_rate = float(lines[1].split(",")[0])

    values = pd.read_csv(
        file,
        skiprows=2,
        header=None,
        names=["x", "y", "z"]
    )

    values = values.apply(
        pd.to_numeric,
        errors="coerce"
    ).dropna()

    timestamps = pd.date_range(
        start=start_time,
        periods=len(values),
        freq=pd.Timedelta(seconds=1 / sample_rate)
    )

    return pd.DataFrame({
        "timestamp": timestamps,
        "acc_x": values["x"].to_numpy(),
        "acc_y": values["y"].to_numpy(),
        "acc_z": values["z"].to_numpy()
    })


def main():

    subject = STRESS / "f01"

    print("=" * 60)
    print("PHYSIONET DATASET TEST")
    print("=" * 60)

    print("\nDataset:")
    print(DATASET)

    print("\nSubject:")
    print(subject)

    if not subject.exists():
        print("\nERROR: f01 folder not found.")
        print("Check the dataset location.")
        return

    print("\nReading HR...")
    hr = read_hr(subject)

    print("Reading IBI...")
    ibi = read_ibi(subject)

    print("Reading ACC...")
    acc = read_acc(subject)

    print("\n" + "=" * 60)
    print("RESULT")
    print("=" * 60)

    print(f"\nHeart-rate samples : {len(hr):,}")
    print(f"IBI samples        : {len(ibi):,}")
    print(f"ACC samples        : {len(acc):,}")

    print("\nHeart-rate:")
    print(hr.head())

    print("\nIBI:")
    print(ibi.head())

    print("\nAccelerometer:")
    print(acc.head())

    print("\nHeart-rate statistics:")
    print(hr["heart_rate"].describe())

    print("\nACC magnitude example:")

    acc["activity"] = np.sqrt(
        acc["acc_x"] ** 2 +
        acc["acc_y"] ** 2 +
        acc["acc_z"] ** 2
    )

    print(acc["activity"].describe())

    print("\nSUCCESS: Real PhysioNet data was read correctly.")


if __name__ == "__main__":
    main()