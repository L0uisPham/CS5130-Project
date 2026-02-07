from __future__ import annotations

import re
from typing import List

import numpy as np
import pandas as pd


def normalize_demographics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    df = df.dropna(subset=["Age"])
    df["Age"] = df["Age"].astype(int)
    df = df[df["Age"] >= 18]

    sex = df["Sex"].astype(str).str.upper().str.strip()
    sex_num = sex.map({"M": 0, "MALE": 0, "F": 1, "FEMALE": 1})
    df["sex_num"] = sex_num
    df = df.dropna(subset=["sex_num"])
    df["sex_num"] = df["sex_num"].astype(int)
    return df


def add_patient_id(df: pd.DataFrame, path_col: str = "Path") -> pd.DataFrame:
    df = df.copy()
    pattern = re.compile(r"patient(\d+)", re.IGNORECASE)
    ids: List[str] = []
    for path in df[path_col].astype(str).tolist():
        match = pattern.search(path)
        ids.append(match.group(0) if match else "unknown")
    df["patient_id"] = ids
    return df


def add_age_bins(
    df: pd.DataFrame,
    age_col: str = "Age",
    start_age: int = 18,
    bin_width: int = 10,
) -> pd.DataFrame:
    df = df.copy()
    max_age = int(df[age_col].max())
    edges = list(range(start_age, max_age + bin_width + 1, bin_width))
    if edges[-1] <= max_age:
        edges.append(edges[-1] + bin_width)
    labels = [f"{left}-{right - 1}" for left, right in zip(edges[:-1], edges[1:])]
    df["age_bin"] = pd.cut(
        df[age_col],
        bins=edges,
        labels=labels,
        right=False,
        include_lowest=True,
    ).astype(str)
    return df


def build_strata(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["strata"] = df["age_bin"].astype(str) + "__" + df["sex_num"].astype(str)
    return df

