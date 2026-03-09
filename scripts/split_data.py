from __future__ import annotations

import re
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


BATCH_SIZE: Final[int] = 32
NUM_WORKERS: Final[int] = 4
PIN_MEMORY: Final[bool] = True
RANDOM_STATE: Final[int] = 42

CHEXPERT_LABELS_14: Final[list[str]] = [
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
]

AGE_COL: Final[str] = "Age"
SEX_COL: Final[str] = "Sex"
PATH_COL: Final[str] = "Path"
ABS_PATH_COL: Final[str] = "abs_path"

SEX_MAP: Final[dict[str, int]] = {
    "M": 0,
    "MALE": 0,
    "F": 1,
    "FEMALE": 1,
}


def extract_patient_id(path_str: str) -> str:
    m = re.search(r"(patient\d+)", path_str)
    return m.group(1) if m else path_str


def make_age_bin_edges(min_age: int, max_age: int, bin_width: int = 10) -> list[int]:
    edges = list(range(min_age, max_age + bin_width, bin_width))
    if len(edges) < 2:
        edges = [min_age, min_age + bin_width]
    return edges


class CheXpertDataset(Dataset):
    def __init__(self, df: pd.DataFrame, labels: list[str], transform) -> None:
        self.df: pd.DataFrame = df.reset_index(drop=True).copy()
        self.labels: list[str] = labels
        self.transform = transform

        self.df[self.labels] = self.df[self.labels].apply(
            pd.to_numeric, errors="coerce"
        )
        self.df[self.labels] = self.df[self.labels].fillna(0.0)
        self.df[self.labels] = self.df[self.labels].replace(-1.0, 0.0)

    def __len__(self) -> int:
        return int(len(self.df))

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img = Image.open(str(row[ABS_PATH_COL])).convert("RGB")
        y = torch.from_numpy(row[self.labels].to_numpy(dtype=np.float32))
        x = self.transform(img)
        return x, y


def main() -> tuple[DataLoader, DataLoader, DataLoader]:
    base_dir = Path.cwd().resolve()
    data_dir = (base_dir / "data").resolve()
    processed_dir = (data_dir / "processed_chexpert").resolve()

    train_csv = (processed_dir / "train_clean.csv").resolve()
    val_csv = (processed_dir / "val_clean.csv").resolve()
    test_csv = (processed_dir / "test_clean.csv").resolve()

    out_dir = (data_dir / "processed_chexpert_age_gender_split").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    df_train: pd.DataFrame = pd.read_csv(train_csv)
    df_val: pd.DataFrame = pd.read_csv(val_csv)
    df_test: pd.DataFrame = pd.read_csv(test_csv)
    df: pd.DataFrame = pd.concat([df_train, df_val, df_test], ignore_index=True)

    df[AGE_COL] = pd.to_numeric(df[AGE_COL], errors="coerce")
    df[SEX_COL] = df[SEX_COL].astype(str).str.strip().str.upper()
    df = df.dropna(subset=[AGE_COL, SEX_COL]).copy()

    df[AGE_COL] = df[AGE_COL].astype(float).astype(int)
    df = df[df[AGE_COL] >= 18].copy()

    df["sex_num"] = df[SEX_COL].map(SEX_MAP)
    df = df.dropna(subset=["sex_num"]).copy()
    df["sex_num"] = df["sex_num"].astype(int)

    df["patient_id"] = df[PATH_COL].astype(str).apply(extract_patient_id)

    min_age = int(df[AGE_COL].min())
    max_age = int(df[AGE_COL].max())
    edges = make_age_bin_edges(min_age=min_age, max_age=max_age, bin_width=10)

    df["age_bin"] = pd.cut(df[AGE_COL], bins=edges, right=False, include_lowest=True)
    df["strata"] = df["age_bin"].astype(str) + "__" + df["sex_num"].astype(str)

    patient_df: pd.DataFrame = (
        df.sort_values(["patient_id"])
        .groupby("patient_id", as_index=False)
        .first()[["patient_id", "strata"]]
        .copy()
    )

    counts = patient_df["strata"].value_counts()
    valid_strata = counts[counts >= 2].index
    patient_df = patient_df[patient_df["strata"].isin(valid_strata)].copy()

    valid_pids = pd.Index(patient_df["patient_id"].to_numpy())
    df = df[df["patient_id"].isin(valid_pids)].copy()

    pid_arr = patient_df["patient_id"].to_numpy()
    strata_arr = patient_df["strata"].to_numpy()

    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    train_idx, temp_idx = next(sss1.split(pid_arr, strata_arr))

    train_pids = pd.Index(pid_arr[train_idx])
    temp_patient_df: pd.DataFrame = patient_df.iloc[temp_idx].copy()

    temp_pid_arr = temp_patient_df["patient_id"].to_numpy()
    temp_strata_arr = temp_patient_df["strata"].to_numpy()

    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=RANDOM_STATE)
    val_idx, test_idx = next(sss2.split(temp_pid_arr, temp_strata_arr))

    val_pids = pd.Index(temp_pid_arr[val_idx])
    test_pids = pd.Index(temp_pid_arr[test_idx])

    split_train: pd.DataFrame = df[df["patient_id"].isin(train_pids)].copy()
    split_val: pd.DataFrame = df[df["patient_id"].isin(val_pids)].copy()
    split_test: pd.DataFrame = df[df["patient_id"].isin(test_pids)].copy()

    split_train.to_csv(out_dir / "train.csv", index=False)
    split_val.to_csv(out_dir / "val.csv", index=False)
    split_test.to_csv(out_dir / "test.csv", index=False)

    train_tfms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    eval_tfms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_ds = CheXpertDataset(
        split_train, labels=CHEXPERT_LABELS_14, transform=train_tfms
    )
    val_ds = CheXpertDataset(split_val, labels=CHEXPERT_LABELS_14, transform=eval_tfms)
    test_ds = CheXpertDataset(
        split_test, labels=CHEXPERT_LABELS_14, transform=eval_tfms
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    main()
