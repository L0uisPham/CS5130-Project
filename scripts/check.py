from pathlib import Path
import pandas as pd

BASE_DIR = Path.cwd().resolve()
DATA_DIR = BASE_DIR / "data" / "processed_chexpert"

TRAIN_CSV = DATA_DIR / "train_clean.csv"
VAL_CSV = DATA_DIR / "val_clean.csv"
TEST_CSV = DATA_DIR / "test_clean.csv"

CHEXPERT_LABELS_14 = [
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


def summarize_split(name: str, df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[CHEXPERT_LABELS_14] = (
        df[CHEXPERT_LABELS_14]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .replace(-1.0, 0.0)
    )

    total = len(df)
    stats = []

    for label in CHEXPERT_LABELS_14:
        positives = int(df[label].sum())
        percent = positives / total * 100
        stats.append((label, positives, percent))

    out = pd.DataFrame(stats, columns=["Label", f"{name}_Positives", f"{name}_%"])
    return out


def main():
    train_df = pd.read_csv(TRAIN_CSV)
    val_df = pd.read_csv(VAL_CSV)
    test_df = pd.read_csv(TEST_CSV)

    train_stats = summarize_split("Train", train_df)
    val_stats = summarize_split("Val", val_df)
    test_stats = summarize_split("Test", test_df)

    merged = train_stats.merge(val_stats, on="Label")
    merged = merged.merge(test_stats, on="Label")

    print("\n=== Label Distribution Across Splits ===\n")
    print(merged.to_string(index=False))

    print("\n=== Labels missing in Val/Test ===\n")
    for _, row in merged.iterrows():
        if row["Val_Positives"] == 0:
            print(f"⚠️  {row['Label']} has 0 positives in VAL")
        if row["Test_Positives"] == 0:
            print(f"⚠️  {row['Label']} has 0 positives in TEST")


if __name__ == "__main__":
    main()
