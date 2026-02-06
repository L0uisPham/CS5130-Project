import pandas as pd
import numpy as np
from typing import List, Tuple
from skmultilearn.model_selection import iterative_train_test_split


def clean_chexpert_labels(
    df: pd.DataFrame,
    labels: List[str],
    uncertain_value: float
) -> pd.DataFrame:
    df = df.copy()

    df[labels] = (
        df[labels]
        .apply(pd.to_numeric, errors="coerce")
        .replace(-1, uncertain_value)
        .fillna(0)
        .astype("float32")
    )

    return df


def iterative_multilabel_split_3way(
    df: pd.DataFrame,
    labels: List[str],
    val_fraction: float,
    test_size: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Performs iterative multilabel stratified train / val / test split.

    Args:
        df: Input dataframe.
        labels: Label columns.
        val_fraction: Fraction (of remaining data) for validation.
        test_size: Absolute number of samples for test set.

    Returns:
        (train_df, val_df, test_df)
    """
    X = df.index.values.reshape(-1, 1)
    y = df[labels].values

    # ---- Split off TEST set first (absolute size) ----
    test_fraction = test_size / len(df)

    X_remain, _, X_test, _ = iterative_train_test_split(
        X, y, test_size=test_fraction
    )

    remain_idx = X_remain.flatten()
    test_idx = X_test.flatten()

    df_remain = df.loc[remain_idx].reset_index(drop=True)
    test_df = df.loc[test_idx].reset_index(drop=True)

    # ---- Split remaining into TRAIN / VAL ----
    X_remain = df_remain.index.values.reshape(-1, 1)
    y_remain = df_remain[labels].values

    X_train, _, X_val, _ = iterative_train_test_split(
        X_remain, y_remain, test_size=val_fraction
    )

    train_df = df_remain.loc[X_train.flatten()].reset_index(drop=True)
    val_df = df_remain.loc[X_val.flatten()].reset_index(drop=True)

    return train_df, val_df, test_df


def print_label_distribution(df: pd.DataFrame, labels: List[str], title: str):
    print(f"\n{title}")
    for label in labels:
        print(f"{label:25s}: {int(df[label].sum())}")


def build_stratified_chexpert_split(
    train_csv: str,
    valid_csv: str,
    train_out: str,
    val_out: str,
    test_out: str,
    labels: List[str],
    val_fraction: float,
    test_size: int
):
    df_train = pd.read_csv(train_csv)
    df_valid = pd.read_csv(valid_csv)

    # ---- Split train into stratified train/val/test ----
    train_df, val_df_strat, test_df = iterative_multilabel_split_3way(
        df_train, labels, val_fraction, test_size
    )

    # ---- Clean train and test ----
    train_df = clean_chexpert_labels(train_df, labels, uncertain_value=0.5)
    test_df  = clean_chexpert_labels(test_df, labels, uncertain_value=0.0)

    # ---- Clean original valid and combine with stratified val ----
    val_df_strat = clean_chexpert_labels(val_df_strat, labels, uncertain_value=0.0)
    df_valid    = clean_chexpert_labels(df_valid, labels, uncertain_value=0.0)
    val_df      = pd.concat([val_df_strat, df_valid], ignore_index=True)

    # ---- Save ----
    train_df.to_csv(train_out, index=False)
    val_df.to_csv(val_out, index=False)
    test_df.to_csv(test_out, index=False)

    print_label_distribution(train_df, labels, "Train label counts")
    print_label_distribution(val_df, labels, "Validation label counts")
    print_label_distribution(test_df, labels, "Test label counts")

    print("\n✓ Stratified train / val / test CSVs saved")
    print(f"Train: {len(train_df)} samples")
    print(f"Val:   {len(val_df)} samples")
    print(f"Test:  {len(test_df)} samples")


# --------------------------------------------------
# Example Usage
# --------------------------------------------------
if __name__ == "__main__":

    TRAIN_CSV = "data/CheXpert-v1.0-small/train.csv"
    VALID_CSV = "data/CheXpert-v1.0-small/valid.csv"
    TRAIN_OUT = "data/CheXpert-v1.0-small/train_strat.csv"
    VAL_OUT = "data/CheXpert-v1.0-small/valid_strat.csv"
    TEST_OUT = "data/CheXpert-v1.0-small/test_strat.csv"

    TEST_SIZE = 234
    VAL_FRACTION = 234 / (223414 - TEST_SIZE)

    LABELS = [
        "Atelectasis", "Cardiomegaly", "Consolidation",
        "Edema", "Enlarged Cardiomediastinum", "Fracture",
        "Lung Lesion", "Lung Opacity", "Pleural Effusion",
        "Pleural Other", "Pneumonia", "Pneumothorax",
        "Support Devices"
    ]

    build_stratified_chexpert_split(
        TRAIN_CSV,
        VALID_CSV,
        TRAIN_OUT,
        VAL_OUT,
        TEST_OUT,
        LABELS,
        VAL_FRACTION,
        TEST_SIZE
    )

"""
OUTPUT

Train label counts
Atelectasis              : 33304
Cardiomegaly             : 26943
Consolidation            : 14739
Edema                    : 52136
Enlarged Cardiomediastinum: 10775
Fracture                 : 9017
Lung Lesion              : 9163
Lung Opacity             : 105359
Pleural Effusion         : 86006
Pleural Other            : 3511
Pneumonia                : 6019
Pneumothorax             : 19407
Support Devices          : 115759

Validation label counts
Atelectasis              : 35
Cardiomegaly             : 29
Consolidation            : 20
Edema                    : 55
Enlarged Cardiomediastinum: 11
Fracture                 : 11
Lung Lesion              : 10
Lung Opacity             : 111
Pleural Effusion         : 90
Pleural Other            : 8
Pneumonia                : 11
Pneumothorax             : 20
Support Devices          : 121

Test label counts
Atelectasis              : 37
Cardiomegaly             : 28
Consolidation            : 24
Edema                    : 55
Enlarged Cardiomediastinum: 12
Fracture                 : 12
Lung Lesion              : 13
Lung Opacity             : 111
Pleural Effusion         : 91
Pleural Other            : 4
Pneumonia                : 9
Pneumothorax             : 21
Support Devices          : 121

✓ Stratified train / val / test CSVs saved
Train: 222946 samples
Val:   234 samples
Test:  234 samples
"""