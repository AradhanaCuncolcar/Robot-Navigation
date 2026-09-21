"""
data_loader.py - Load the Kaggle "Wall-Following Robot" archive.

  sensor_readings_4.csv   4 simplified distances + class
  sensor_readings_2.csv   front + left distances + class
  sensor_readings_24.csv  24 raw ultrasound readings + class
Handles files with or without a header row and Windows line endings.
"""
import os
import numpy as np
import pandas as pd
import config


def _read(path, n_cols):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Unzip the Kaggle archive into the data/ folder.")
    with open(path, "r") as f:
        first = f.readline().strip().split(",")
    try:
        float(first[0]); header = None
    except ValueError:
        header = 0
    df = pd.read_csv(path, header=header)
    if df.shape[1] != n_cols:
        raise ValueError(f"{os.path.basename(path)}: expected {n_cols} columns, found {df.shape[1]}")
    return df


def load_dataset(variant=config.DEFAULT_VARIANT, path=None) -> pd.DataFrame:
    v = config.VARIANTS[variant]
    path = path or os.path.join(config.DATA_DIR, v["file"])
    df = _read(path, len(v["sensors"]) + 1)
    df.columns = v["sensors"] + ["Class"]
    df["Class"] = df["Class"].astype(str).str.strip()
    df[v["sensors"]] = df[v["sensors"]].astype(float)
    unknown = set(df["Class"]) - set(config.ACTIONS)
    if unknown:
        raise ValueError(f"Unknown class labels: {unknown}")
    return df.reset_index(drop=True)


def load_raw24(path=None) -> pd.DataFrame:
    path = path or os.path.join(config.DATA_DIR, config.RAW24_FILE)
    df = _read(path, 25)
    df.columns = [f"US{i}" for i in range(1, 25)] + ["Class"]
    df["Class"] = df["Class"].astype(str).str.strip()
    return df


def reduce_24_to_4(raw: pd.DataFrame) -> pd.DataFrame:
    X = raw.iloc[:, :24].to_numpy(float)
    return pd.DataFrame({s: X[:, cols].min(axis=1) for s, cols in config.ARC_COLUMNS_24.items()})


def verify_24_to_4():
    """Fraction of rows where min-over-arcs of the raw sonars equals the 4-sensor file."""
    red = reduce_24_to_4(load_raw24())
    four = load_dataset("4")
    return {s: float(np.mean(np.isclose(red[s], four[s]))) for s in config.ALL_SENSORS}


if __name__ == "__main__":
    for k in config.VARIANTS:
        d = load_dataset(k)
        print(f"variant {k}: {len(d)} rows, columns {list(d.columns)}")
    print("Class distribution:\n", load_dataset()["Class"].value_counts())
    print("24 -> 4 reduction match:", verify_24_to_4())
