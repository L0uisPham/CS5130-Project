from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.core.paths import get_project_root
from src.core.types import Batch
from src.datasets.chexpert_dataset import CheXpertDataset
from src.datasets.demographics import (
    add_age_bins,
    add_patient_id,
    build_strata,
    normalize_demographics,
)
from src.datasets.transforms import build_transforms
from src.datasets.splitters import ProcessSplit


def _prepare_dataframe(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "Age" not in df.columns or "Sex" not in df.columns:
        raise ValueError("CSV must include Age and Sex columns for demographics.")
    df = normalize_demographics(df)
    df = add_patient_id(df, path_col="Path")
    df = add_age_bins(df, age_col="Age", start_age=18, bin_width=10)
    df = build_strata(df)
    return df


def _collate_fn(items):
    xs = torch.stack([item.x for item in items], dim=0)
    ys = torch.stack([item.y for item in items], dim=0)
    sex_num = torch.tensor([item.meta["sex_num"] for item in items], dtype=torch.int64)
    age = torch.tensor([item.meta["age"] for item in items], dtype=torch.int64)
    age_bin = [item.meta["age_bin"] for item in items]
    patient_id = [item.meta["patient_id"] for item in items]
    path = [item.meta["path"] for item in items]
    return Batch(
        x=xs,
        y=ys,
        meta={
            "sex_num": sex_num,
            "age": age,
            "age_bin": age_bin,
            "patient_id": patient_id,
            "path": path,
        },
    )


def build_dataloaders(cfg) -> Dict[str, DataLoader]:
    data_cfg = cfg.get("data", {})
    root = get_project_root()

    train_csv = root / data_cfg["train_csv"]
    val_csv = root / data_cfg["val_csv"]
    test_csv = root / data_cfg["test_csv"]
    root_dir = root / data_cfg.get("root_dir", "data")

    if bool(data_cfg.get("generate_stratified_splits", False)):
        splitter = ProcessSplit(
            train_csv=train_csv,
            valid_csv=val_csv,
            output_dir=data_cfg.get("split_output_dir", root / "data/processed_chexpert"),
        )
        splitter.run()
        train_csv = Path(data_cfg.get("split_output_dir", root / "data/processed_chexpert")) / "train_strat.csv"
        val_csv = Path(data_cfg.get("split_output_dir", root / "data/processed_chexpert")) / "valid_strat.csv"
        test_csv = Path(data_cfg.get("split_output_dir", root / "data/processed_chexpert")) / "test_strat.csv"

    label_names = cfg.get("labels", [])
    if not label_names:
        raise ValueError("Config must provide labels list.")

    train_df = _prepare_dataframe(train_csv)
    val_df = _prepare_dataframe(val_csv)
    test_df = _prepare_dataframe(test_csv)

    train_tfms, eval_tfms = build_transforms(cfg)

    train_dataset = CheXpertDataset(
        csv_path=train_csv,
        root_dir=root_dir,
        label_names=label_names,
        transform=train_tfms,
        dataframe=train_df,
    )
    val_dataset = CheXpertDataset(
        csv_path=val_csv,
        root_dir=root_dir,
        label_names=label_names,
        transform=eval_tfms,
        dataframe=val_df,
    )
    test_dataset = CheXpertDataset(
        csv_path=test_csv,
        root_dir=root_dir,
        label_names=label_names,
        transform=eval_tfms,
        dataframe=test_df,
    )

    batch_size = int(data_cfg.get("batch_size", 32))
    num_workers = int(data_cfg.get("num_workers", 4))
    pin_memory = bool(data_cfg.get("pin_memory", True))
    prefetch_factor = int(data_cfg.get("prefetch_factor", 2))
    persistent_workers = bool(data_cfg.get("persistent_workers", num_workers > 0))

    return {
        "train": DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            persistent_workers=persistent_workers if num_workers > 0 else False,
            collate_fn=_collate_fn,
        ),
        "val": DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            persistent_workers=persistent_workers if num_workers > 0 else False,
            collate_fn=_collate_fn,
        ),
        "test": DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            persistent_workers=persistent_workers if num_workers > 0 else False,
            collate_fn=_collate_fn,
        ),
    }
