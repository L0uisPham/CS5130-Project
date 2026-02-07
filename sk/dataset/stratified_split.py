import pandas as pd
import numpy as np
from skmultilearn.model_selection import iterative_train_test_split
from pathlib import Path


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
        # optionally fill Age NaN with median
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

        # -----------------------
        # Pull 234 stratified samples FROM TRAIN → VALID
        # -----------------------
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

        # -----------------------
        # Create TEST split (468) FROM remaining train
        # -----------------------
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

        def print_label_distribution(df: pd.DataFrame, labels: list, title: str):
            print(f"\n{title}")
            for label in labels:
                print(f"{label:26s}: {int(df[label].sum())}")

        train_strat.to_csv(self.output_dir / "train_strat.csv", index=False)
        valid_strat.to_csv(self.output_dir / "valid_strat.csv", index=False)
        test_strat.to_csv(self.output_dir / "test_strat.csv", index=False)

        # Print label distributions
        label_cols = strat_cols  # LABEL_COLS + AGE_LABELS

        print_label_distribution(train_strat, label_cols, "Train label counts")
        print_label_distribution(
            valid_strat, label_cols, "Validation label counts")
        print_label_distribution(test_strat, label_cols, "Test label counts")

        # Print sample counts
        print("\nStratified train / val / test CSVs saved")
        print(f"Train: {len(train_strat)} samples")
        print(f"Valid:   {len(valid_strat)} samples")
        print(f"Test :  {len(test_strat)} samples")


def main():
    splitter = ProcessSplit(
        train_csv="data/CheXpert-v1.0-small/train.csv",
        valid_csv="data/CheXpert-v1.0-small/valid.csv",
        output_dir="data/CheXpert-v1.0-small",
    )
    splitter.run()


if __name__ == "__main__":
    main()


"""
OUTPUT

Train label counts
No Finding               : 22311
Enlarged Cardiomediastinum: 10762
Cardiomegaly             : 26907
Lung Opacity             : 105244
Lung Lesion              : 9152
Edema                    : 52080
Consolidation            : 14729
Pneumonia                : 6016
Atelectasis              : 33257
Pneumothorax             : 19377
Pleural Effusion         : 85906
Pleural Other            : 3511
Fracture                 : 9005
Support Devices          : 115629
age_0_29                 : 15286
age_30_44                : 27028
age_45_59                : 58006
age_60_74                : 68087
age_75_plus              : 54275

Validation label counts
No Finding               : 61
Enlarged Cardiomediastinum: 121
Cardiomegaly             : 96
Lung Opacity             : 236
Lung Lesion              : 14
Edema                    : 101
Consolidation            : 54
Pneumonia                : 17
Atelectasis              : 122
Pneumothorax             : 34
Pleural Effusion         : 167
Pleural Other            : 4
Fracture                 : 16
Support Devices          : 233
age_0_29                 : 41
age_30_44                : 54
age_45_59                : 124
age_60_74                : 143
age_75_plus              : 124

Test label counts
No Finding               : 47
Enlarged Cardiomediastinum: 24
Cardiomegaly             : 65
Lung Opacity             : 227
Lung Lesion              : 21
Edema                    : 110
Consolidation            : 33
Pneumonia                : 14
Atelectasis              : 77
Pneumothorax             : 45
Pleural Effusion         : 181
Pleural Other            : 9
Fracture                 : 19
Support Devices          : 246
age_0_29                 : 35
age_30_44                : 61
age_45_59                : 122
age_60_74                : 148
age_75_plus              : 114

Stratified train / val / test CSVs saved
Train: 222682 samples
Val:   486 samples
Test:  480 samples
"""
