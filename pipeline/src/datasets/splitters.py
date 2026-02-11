from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from skmultilearn.model_selection import iterative_train_test_split


class ProcessSplit:
    LABEL_COLS = [
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

    AGE_BINS = [0, 30, 45, 60, 75, np.inf]
    AGE_LABELS = [
        "age_0_29",
        "age_30_44",
        "age_45_59",
        "age_60_74",
        "age_75_plus",
    ]

    def __init__(self, train_csv, valid_csv, output_dir="data/processed"):
        self.train_csv = Path(train_csv)
        self.valid_csv = Path(valid_csv)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _load_and_clean(self, csv_path):
        df = pd.read_csv(csv_path).copy()
        df_labels = df[self.LABEL_COLS].copy()
        df_labels = df_labels.replace(-1, 0)
        df[self.LABEL_COLS] = df_labels
        if "Age" in df.columns:
            df["Age"] = df["Age"].fillna(df["Age"].median())
        return df

    def _add_age_bins(self, df):
        age_bins = pd.cut(
            df["Age"],
            bins=self.AGE_BINS,
            labels=self.AGE_LABELS,
            right=False,
        )
        age_ohe = pd.get_dummies(age_bins)
        return pd.concat([df, age_ohe], axis=1)

    def run(self):
        train_df = self._load_and_clean(self.train_csv)
        valid_df = self._load_and_clean(self.valid_csv)

        train_df = self._add_age_bins(train_df)
        valid_df = self._add_age_bins(valid_df)

        strat_cols = self.LABEL_COLS + self.AGE_LABELS

        X = train_df.drop(columns=strat_cols)
        y = train_df[strat_cols]

        n_add = 234
        test_frac = n_add / len(train_df)

        X_rem, y_rem, X_add, y_add = iterative_train_test_split(
            X.values,
            y=y.fillna(0).astype(np.int8).values,
            test_size=test_frac,
        )

        train_rem = pd.concat(
            [
                pd.DataFrame(X_rem, columns=X.columns),
                pd.DataFrame(y_rem, columns=strat_cols),
            ],
            axis=1,
        )

        valid_add = pd.concat(
            [
                pd.DataFrame(X_add, columns=X.columns),
                pd.DataFrame(y_add, columns=strat_cols),
            ],
            axis=1,
        )

        valid_strat = pd.concat([valid_df, valid_add]).reset_index(drop=True)

        X2 = train_rem.drop(columns=strat_cols)
        y2 = train_rem[strat_cols]

        n_test = 468
        test_frac = n_test / len(train_rem)

        X_train, y_train, X_test, y_test = iterative_train_test_split(
            X2.values,
            y2.fillna(0).astype(np.int8).values,
            test_size=test_frac,
        )

        train_strat = pd.concat(
            [
                pd.DataFrame(X_train, columns=X2.columns),
                pd.DataFrame(y_train, columns=strat_cols),
            ],
            axis=1,
        )

        test_strat = pd.concat(
            [
                pd.DataFrame(X_test, columns=X2.columns),
                pd.DataFrame(y_test, columns=strat_cols),
            ],
            axis=1,
        )

        train_strat.to_csv(self.output_dir / "train_strat.csv", index=False)
        valid_strat.to_csv(self.output_dir / "valid_strat.csv", index=False)
        test_strat.to_csv(self.output_dir / "test_strat.csv", index=False)
