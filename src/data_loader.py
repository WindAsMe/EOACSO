"""Loaders for the Parkinson's speech datasets used in this project.

Oxford dataset (Little et al. 2008): 195 voice recordings from 31 subjects,
22 numeric features, binary target `status` (1 = PD, 0 = healthy). Subject id
is embedded in the `name` field (e.g. ``phon_R01_S01_1`` -> subject ``S01``),
so a GroupKFold split by subject is required to avoid leakage.

Naranjo dataset (UCI "Parkinson Dataset with Replicated Acoustic Features",
ID 489): 80 subjects (40 PD + 40 healthy, perfectly class-balanced), 3
repeated recordings each = 240 rows, 45 features. Also the "small" PD
dataset used in the mHGS baseline paper (Hashim et al. 2023). Subject id is
the `ID` column (e.g. ``CONT-01``), so grouping is required, same as Oxford.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


@dataclass
class Dataset:
    name: str
    X: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    feature_names: list


def load_oxford() -> Dataset:
    df = pd.read_csv(RAW_DIR / "parkinsons.data")
    groups = df["name"].str.extract(r"(S\d+)")[0].to_numpy()
    y = df["status"].to_numpy(dtype=int)
    feature_names = [c for c in df.columns if c not in ("name", "status")]
    X = df[feature_names].to_numpy(dtype=float)
    return Dataset(name="oxford", X=X, y=y, groups=groups, feature_names=feature_names)


def load_naranjo() -> Dataset:
    df = pd.read_csv(RAW_DIR / "ReplicatedAcousticFeatures-ParkinsonDatabase.csv")
    y = df["Status"].to_numpy(dtype=int)
    groups = df["ID"].to_numpy()
    feature_names = [c for c in df.columns if c not in ("ID", "Recording", "Status")]
    X = df[feature_names].to_numpy(dtype=float)
    return Dataset(name="naranjo", X=X, y=y, groups=groups, feature_names=feature_names)


DATASETS = {"oxford": load_oxford, "naranjo": load_naranjo}


def load(name: str) -> Dataset:
    if name not in DATASETS:
        raise ValueError(f"unknown dataset {name!r}, choose from {list(DATASETS)}")
    return DATASETS[name]()


if __name__ == "__main__":
    for key in DATASETS:
        ds = load(key)
        n_pos = int(ds.y.sum())
        print(f"{ds.name}: X={ds.X.shape} y_pos={n_pos}/{len(ds.y)} groups={len(set(ds.groups))}")
