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

    SEX_LABELS = [
        "sex_male",
        "sex_female"
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

    def _add_sex_ohe(self, df):
        sex = (
            df["Sex"]
            .astype(str)
            .str.strip()
            .str.lower()
            .map({
                "male": "sex_male",
                "female": "sex_female",
            })
        )

        sex_ohe = pd.get_dummies(sex)
        sex_ohe = sex_ohe.reindex(columns=self.SEX_LABELS, fill_value=0)

        return pd.concat([df, sex_ohe], axis=1)

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

        train_df = self._add_sex_ohe(train_df)
        valid_df = self._add_sex_ohe(valid_df)

        train_df = self._add_age_bins(train_df)
        valid_df = self._add_age_bins(valid_df)

        strat_cols = self.LABEL_COLS + self.SEX_LABELS + self.AGE_LABELS

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

        def print_label_distribution(df: pd.DataFrame, labels: list, title: str):
            print(f"\n{title}")
            for label in labels:
                print(f"{label:26s}: {int(df[label].sum())}")

        train_strat.to_csv(self.output_dir / "train_strat.csv", index=False)
        valid_strat.to_csv(self.output_dir / "valid_strat.csv", index=False)
        test_strat.to_csv(self.output_dir / "test_strat.csv", index=False)

        print_label_distribution(train_strat, strat_cols, "Train label counts")
        print_label_distribution(
            valid_strat, strat_cols, "Validation label counts")
        print_label_distribution(test_strat, strat_cols, "Test label counts")

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
No Finding                : 22310
Enlarged Cardiomediastinum: 10764
Cardiomegaly              : 26906
Lung Opacity              : 105240
Lung Lesion               : 9156
Edema                     : 52076
Consolidation             : 14734
Pneumonia                 : 6024
Atelectasis               : 33261
Pneumothorax              : 19379
Pleural Effusion          : 85889
Pleural Other             : 3500
Fracture                  : 9005
Support Devices           : 115627
sex_male                  : 132217
sex_female                : 90464
age_0_29                  : 15287
age_30_44                 : 27030
age_45_59                 : 57999
age_60_74                 : 68093
age_75_plus               : 54272

Validation label counts
No Finding                : 62
Enlarged Cardiomediastinum: 123
Cardiomegaly              : 96
Lung Opacity              : 239
Lung Lesion               : 12
Edema                     : 101
Consolidation             : 50
Pneumonia                 : 14
Atelectasis               : 120
Pneumothorax              : 33
Pleural Effusion          : 168
Pleural Other             : 6
Fracture                  : 11
Support Devices           : 235
sex_male                  : 269
sex_female                : 207
age_0_29                  : 37
age_30_44                 : 52
age_45_59                 : 127
age_60_74                 : 142
age_75_plus               : 118

Test label counts
No Finding                : 47
Enlarged Cardiomediastinum: 20
Cardiomegaly              : 66
Lung Opacity              : 228
Lung Lesion               : 19
Edema                     : 114
Consolidation             : 32
Pneumonia                 : 9
Atelectasis               : 75
Pneumothorax              : 44
Pleural Effusion          : 197
Pleural Other             : 18
Fracture                  : 24
Support Devices           : 246
sex_male                  : 278
sex_female                : 212
age_0_29                  : 38
age_30_44                 : 61
age_45_59                 : 126
age_60_74                 : 143
age_75_plus               : 123

Stratified train / val / test CSVs saved
Train: 222681 samples
Valid:   476 samples
Test :  491 samples
"""
