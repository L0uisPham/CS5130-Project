import pandas as pd
import numpy as np
from typing import List, Tuple
from skmultilearn.model_selection import iterative_train_test_split


def clean_chexpert_labels(df: pd.DataFrame, labels: List[str]) -> pd.DataFrame:
    """
    Cleans CheXpert label columns.

    - Converts labels to numeric
    - Replaces uncertain (-1) with 0
    - Fills missing values with 0
    - Casts to integer

    Args:
        df: Input dataframe containing CheXpert labels.
        labels: List of label column names.

    Returns:
        pd.DataFrame: DataFrame with cleaned labels.
    """
    df = df.copy()

    df[labels] = (
        df[labels]
        .apply(pd.to_numeric, errors="coerce")
        .replace(-1, 0)
        .fillna(0)
        .astype(int)
    )

    return df


def iterative_multilabel_split(
    df: pd.DataFrame,
    labels: List[str],
    val_fraction: float
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Performs iterative multilabel stratified split.

    Args:
        df: Input dataframe.
        labels: List of label columns.
        val_fraction: Fraction of samples to use for validation.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (train_df, val_df)
    """
    X = df.index.values.reshape(-1, 1)
    y = df[labels].values

    X_train, _, X_val, _ = iterative_train_test_split(
        X, y, test_size=val_fraction
    )

    train_idx = X_train.flatten()
    val_idx = X_val.flatten()

    train_df = df.loc[train_idx].reset_index(drop=True)
    val_df = df.loc[val_idx].reset_index(drop=True)

    return train_df, val_df


def print_label_distribution(df: pd.DataFrame, labels: List[str], title: str):
    """
    Prints positive counts for each label.

    Args:
        df: DataFrame to analyze.
        labels: Label column names.
        title: Title for the output block.
    """
    print(f"\n{title}")
    for label in labels:
        count = int(df[label].sum())
        print(f"{label:25s}: {count}")


def build_stratified_chexpert_split(
    input_csv: str,
    train_out: str,
    val_out: str,
    labels: List[str],
    val_fraction: float
):
    """
    Builds stratified CheXpert train/validation splits and saves them to CSV.

    Args:
        input_csv: Path to input CheXpert CSV.
        train_out: Output path for stratified training CSV.
        val_out: Output path for stratified validation CSV.
        labels: Label column names.
        val_fraction: Fraction of dataset to use as validation.
    """
    # Load dataset
    df = pd.read_csv(input_csv)

    # Clean labels
    df = clean_chexpert_labels(df, labels)

    # Split
    train_df, val_df = iterative_multilabel_split(df, labels, val_fraction)

    # Sanity check
    print_label_distribution(val_df, labels, "Validation label counts")

    # Save CSVs
    train_df.to_csv(train_out, index=False)
    val_df.to_csv(val_out, index=False)

    print("\n✓ Stratified train/val CSVs saved")
    print(f"Train: {len(train_df)} samples")
    print(f"Val:   {len(val_df)} samples")


# --------------------------------------------------
# Example Usage
# --------------------------------------------------
if __name__ == "__main__":

    INPUT_CSV = "data/CheXpert-v1.0-small/train.csv"
    TRAIN_OUT = "data/CheXpert-v1.0-small/train_strat.csv"
    VAL_OUT = "data/CheXpert-v1.0-small/valid_strat.csv"

    VAL_FRACTION = 234 / 223414

    LABELS = [
        "Atelectasis", "Cardiomegaly", "Consolidation",
        "Edema", "Enlarged Cardiomediastinum", "Fracture",
        "Lung Lesion", "Lung Opacity", "Pleural Effusion",
        "Pleural Other", "Pneumonia", "Pneumothorax",
        "Support Devices"
    ]

    build_stratified_chexpert_split(
        INPUT_CSV,
        TRAIN_OUT,
        VAL_OUT,
        LABELS,
        VAL_FRACTION
    )
