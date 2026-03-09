from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.core.types import Batch


class CheXpertDataset(Dataset):
    def __init__(
        self,
        csv_path: Path,
        root_dir: Path,
        label_names: List[str],
        transform=None,
        dataframe: Optional[pd.DataFrame] = None,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.root_dir = Path(root_dir)
        self.label_names = label_names
        self.transform = transform

        if dataframe is None:
            df = pd.read_csv(self.csv_path)
        else:
            df = dataframe.copy()

        df[self.label_names] = (
            df[self.label_names]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0.0)
            .replace(-1, 0)
            .astype("float32")
        )
        self.df = df.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Batch:
        row = self.df.iloc[idx]
        rel_path = str(row["Path"])
        path = Path(rel_path)
        if not path.is_absolute():
            path = self.root_dir / rel_path
        abs_path = path.resolve()

        image = Image.open(abs_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        label_series = pd.to_numeric(row[self.label_names], errors="coerce").fillna(0.0)
        y = torch.tensor(label_series.to_numpy(dtype="float32"), dtype=torch.float32)
        meta = {
            "sex_num": int(row["sex_num"]),
            "age": int(row["Age"]),
            "age_bin": str(row["age_bin"]),
            "patient_id": str(row["patient_id"]),
            "path": str(abs_path),
        }
        return Batch(x=image, y=y, meta=meta)

    @property
    def num_classes(self) -> int:
        return len(self.label_names)
