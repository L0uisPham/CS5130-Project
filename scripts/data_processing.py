import os, re, json, random
from pathlib import Path
import numpy as np
import pandas as pd

CHEXPERT_LABELS_14 = [
    "No Finding","Enlarged Cardiomediastinum","Cardiomegaly","Lung Opacity","Lung Lesion",
    "Edema","Consolidation","Pneumonia","Atelectasis","Pneumothorax","Pleural Effusion",
    "Pleural Other","Fracture","Support Devices",
]

#Extract the id using folder's names, regex shenanigans 
def extract_patient_id(path_str: str) -> str:
    m = re.search(r"(patient\d+)", str(path_str))
    return m.group(1) if m else str(path_str)

#Add more pd columns for ease of reference 
def resolve_paths(df: pd.DataFrame, data_root: Path) -> pd.DataFrame:
    df = df.copy()
    def to_abs(p: str) -> str:
        p = str(p).replace("\\", "/")
        p = re.sub(r"^CheXpert-v1\.0-small/", "", p)
        return str((data_root / p).resolve())
    df["abs_path"] = df["Path"].apply(to_abs)
    df["exists"] = df["abs_path"].apply(lambda p: Path(p).exists())
    df["patient_id"] = df["Path"].apply(extract_patient_id)
    return df

#Many ML pipeliens can only process binary values, therefore this will set all uncertain values(-1) to either 0 or 1, changable through parameter
def apply_uncertain_policy(df: pd.DataFrame, labels, policy: str) -> pd.DataFrame:
    df = df.copy()
    for c in labels:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if policy == "uzeros":
        df[labels] = df[labels].replace(-1, 0).fillna(0)
    elif policy == "uones":
        df[labels] = df[labels].replace(-1, 1).fillna(0)
    else:
        raise ValueError("policy must be 'uzeros' or 'uones'")
    return df

#Split by patient to prevent leakage over from train to val/test which might lead to very high accuracy as it is running on trainned dataset
def patient_split(df: pd.DataFrame, seed=1337, test_frac=0.1, val_frac=0.1):
    pats = df["patient_id"].unique().tolist()
    rng = np.random.default_rng(seed)
    rng.shuffle(pats)
    n_test = int(round(test_frac * len(pats)))
    test_pats = set(pats[:n_test])
    rem = pats[n_test:]
    n_val = int(round(val_frac * len(rem)))
    val_pats = set(rem[:n_val])
    train_pats = set(rem[n_val:])
    train = df[df["patient_id"].isin(train_pats)].reset_index(drop=True)
    val = df[df["patient_id"].isin(val_pats)].reset_index(drop=True)
    test = df[df["patient_id"].isin(test_pats)].reset_index(drop=True)
    return train, val, test

def main():
    data_root = Path("data/CheXpert-v1.0-small").resolve()
    out_dir = Path("data/processed_chexpert").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = CHEXPERT_LABELS_14
    policy = "uzeros" 
    seed = 1337

    train_df = pd.read_csv(data_root / "train.csv")
    valid_df = pd.read_csv(data_root / "valid.csv")  

    train_df = resolve_paths(train_df, data_root)
    valid_df = resolve_paths(valid_df, data_root)

    missing = int((~train_df["exists"]).sum())
    if missing:
        print(f"WARNING: {missing} missing train images (check paths).")

    train_df = apply_uncertain_policy(train_df, labels, policy)
    valid_df = apply_uncertain_policy(valid_df, labels, policy)  

    #80, 10, 10 split here
    tr, va, te = patient_split(train_df, seed=seed, test_frac=0.10, val_frac=0.10)

    tr.to_csv(out_dir / "train_clean.csv", index=False)
    va.to_csv(out_dir / "val_clean.csv", index=False)
    te.to_csv(out_dir / "test_clean.csv", index=False)
    valid_df.to_csv(out_dir / "valid_official.csv", index=False)

    report = {
        "policy": policy,
        "seed": seed,
        "counts": {
            "train_clean": len(tr),
            "val_clean": len(va),
            "test_clean": len(te),
            "valid_official": len(valid_df),
        },
        "missing_train_images": missing,
    }
    (out_dir / "reports").mkdir(exist_ok=True)
    with open(out_dir / "reports" / "prep_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print("Saved cleaned splits to:", out_dir)

if __name__ == "__main__":
    main()
