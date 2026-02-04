"""
chex_all_in_one_train.py

One-file pipeline:
1) (Optional) Download + unzip CheXpert-v1.0-small.zip (Stanford)
2) Process CheXlocalize-format data:
   - val_labels.csv + val/ images
   - test_labels.csv + test/ images
   - build abs_path, apply uncertain policy, split 80/20 from val
3) Provide PyTorch Dataset + DataLoaders
4) Train ResNet34 and EfficientNet-B4 (multi-label) + basic eval metrics

USAGE (PowerShell):
  python chex_all_in_one_train.py --mode prep
  python chex_all_in_one_train.py --mode smoke
  python chex_all_in_one_train.py --mode train
  python chex_all_in_one_train.py --mode all

Edit CHEXLOCALIZE_ROOT below to your real path.
"""

import sys
import json
import zipfile
import argparse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms, models


# ----------------------------
# Labels (CheXpert 14)
# ----------------------------
# These are the standard 14 CheXpert pathologies/labels used in many CXR tasks.
# Your CSVs must contain columns with these exact names.
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

# Optional: Stanford-hosted "small" CheXpert zip (used only if user runs --mode download_chexpert)
CHEXPERT_ZIP_URL = "http://download.cs.stanford.edu/deep/CheXpert-v1.0-small.zip"


# ============================================================
# A) (Optional) Download + unzip CheXpert-v1.0-small
# ============================================================

def download_file(url: str, out_path: Path):
    """
    Download a file from `url` to `out_path` with a console progress bar.
    """
    # Create parent folders if they don't exist
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def progress(block_num, block_size, total_size):
        """
        `reporthook` for urllib that prints progress.
        """
        downloaded = block_num * block_size
        percent = downloaded / total_size * 100 if total_size > 0 else 0
        mb_done = downloaded / (1024 * 1024)
        mb_total = total_size / (1024 * 1024) if total_size > 0 else 0

        sys.stdout.write(
            f"\rDownloading: {percent:6.2f}% ({mb_done:8.1f} MB / {mb_total:8.1f} MB)"
        )
        sys.stdout.flush()

    print(f"Downloading to: {out_path}")
    urllib.request.urlretrieve(url, out_path, reporthook=progress)
    print("\nDownload complete")


def unzip_file(zip_path: Path, extract_to: Path):
    """
    Unzip `zip_path` into `extract_to.parent`.
    Note: Using extract_to.parent keeps the default folder structure of the zip.
    """
    print(f"Extracting {zip_path} ...")
    extract_to.parent.mkdir(parents=True, exist_ok=True)

    # ZipFile handles reading the archive. extractall(...) writes files to disk.
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_to.parent)

    print("Extraction complete")


def ensure_chexpert_downloaded_and_extracted(project_root: Path):
    """
    Convenience wrapper:
    - Download CheXpert-v1.0-small.zip if missing
    - Unzip into project_root/data/CheXpert-v1.0-small if not already extracted
    """
    data_dir = project_root / "data"
    zip_path = data_dir / "CheXpert-v1.0-small.zip"
    extract_dir = data_dir / "CheXpert-v1.0-small"

    # Only download if the zip isn't already on disk
    if not zip_path.exists():
        download_file(CHEXPERT_ZIP_URL, zip_path)

    # Skip unzip if folder exists and isn't empty (helps avoid long re-extractions)
    if extract_dir.exists() and any(extract_dir.iterdir()):
        print(f"Already extracted at {extract_dir}, skipping unzip.")
    else:
        unzip_file(zip_path, extract_dir)

    print(f"CheXpert dataset at: {extract_dir}")
    return extract_dir


# ============================================================
# B) FAST path join for CheXlocalize val/test folders
# ============================================================

def build_filename_index(img_root: Path):
    """
    Build a dict: {filename -> absolute_path_string} by scanning `img_root` once.

    Why:
    - The naive approach `rglob` per-row is extremely slow.
    - Instead, we scan once, then map basenames to full paths in O(N).
    """
    idx = {}
    for p in img_root.rglob("*"):
        if p.is_file():
            # Note: if there are duplicate filenames in different subfolders,
            # later ones will overwrite earlier ones (rare in this dataset).
            idx[p.name] = str(p.resolve())
    return idx


def attach_paths_fast(df: pd.DataFrame, img_root: Path) -> pd.DataFrame:
    """
    Add two columns to df:
    - abs_path: resolved absolute image path ("" if not found)
    - exists: boolean indicating file exists on disk

    Uses a one-time filename index for speed.
    """
    if "Path" not in df.columns:
        raise ValueError("CSV must contain a 'Path' column.")

    df = df.copy()

    # Build lookup: filename -> full path
    idx = build_filename_index(img_root)

    # Convert "Path" entries into just the basename (handles Windows vs Unix slashes)
    basenames = df["Path"].astype(str).apply(lambda s: Path(s.replace("\\", "/")).name)

    # Map basename to absolute path (empty string if missing)
    df["abs_path"] = basenames.map(idx).fillna("")

    # exists=True only if abs_path is non-empty AND file exists
    df["exists"] = df["abs_path"].apply(lambda p: Path(p).exists() if p else False)
    return df


def apply_uncertain_policy(df: pd.DataFrame, labels, policy: str) -> pd.DataFrame:
    """
    Handle CheXpert uncertain labels (-1).

    policy:
      - 'uzeros': treat -1 as 0
      - 'uones' : treat -1 as 1

    Also fills NaNs as 0.
    """
    df = df.copy()

    # Ensure label columns exist and are numeric
    for c in labels:
        if c not in df.columns:
            raise ValueError(f"Missing label column in CSV: '{c}'")
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Replace uncertain values (-1) based on selected policy
    if policy == "uzeros":
        df[labels] = df[labels].replace(-1, 0).fillna(0)
    elif policy == "uones":
        df[labels] = df[labels].replace(-1, 1).fillna(0)
    else:
        raise ValueError("policy must be 'uzeros' or 'uones'")

    return df


def split_train_val_from_val(df_val: pd.DataFrame, train_frac=0.80, seed=1337):
    """
    Randomly split the provided VAL dataframe into train/val subsets.

    Note: This splits ONLY the provided df_val (CheXlocalize has val/test folders).
    """
    df_val = df_val.copy().reset_index(drop=True)

    rng = np.random.default_rng(seed)
    idx = np.arange(len(df_val))
    rng.shuffle(idx)

    # Compute number of train samples from val
    n_train = int(round(train_frac * len(df_val)))

    train_idx = idx[:n_train]
    val_idx = idx[n_train:]

    train_df = df_val.iloc[train_idx].reset_index(drop=True)
    val_df = df_val.iloc[val_idx].reset_index(drop=True)
    return train_df, val_df


def prep_chexlocalize(chexlocalize_root: Path, out_dir: Path, policy="uzeros", seed=1337, train_frac=0.80):
    """
    Preprocess CheXlocalize-style dataset directory:

    Expected structure:
      chexlocalize_root/
        val/
        test/
        val_labels.csv
        test_labels.csv

    Output:
      out_dir/train_clean.csv
      out_dir/val_clean.csv
      out_dir/test_clean.csv
      out_dir/prep_report.json
    """
    VAL_IMG_DIR = chexlocalize_root / "val"
    TEST_IMG_DIR = chexlocalize_root / "test"
    VAL_CSV = chexlocalize_root / "val_labels.csv"
    TEST_CSV = chexlocalize_root / "test_labels.csv"

    # Validate required files/folders exist
    if not VAL_IMG_DIR.exists():
        raise FileNotFoundError(f"Missing folder: {VAL_IMG_DIR}")
    if not TEST_IMG_DIR.exists():
        raise FileNotFoundError(f"Missing folder: {TEST_IMG_DIR}")
    if not VAL_CSV.exists():
        raise FileNotFoundError(f"Missing file: {VAL_CSV}")
    if not TEST_CSV.exists():
        raise FileNotFoundError(f"Missing file: {TEST_CSV}")

    out_dir.mkdir(parents=True, exist_ok=True)
    labels = CHEXPERT_LABELS_14

    # Load CSV label files
    print("Reading CSVs...")
    val_df = pd.read_csv(VAL_CSV)
    test_df = pd.read_csv(TEST_CSV)

    # Attach abs paths (fast scan once)
    print("Indexing val image folder (one-time scan)...")
    val_df = attach_paths_fast(val_df, VAL_IMG_DIR)
    print("Indexing test image folder (one-time scan)...")
    test_df = attach_paths_fast(test_df, TEST_IMG_DIR)

    # Report missing image files if any paths did not resolve
    missing_val = int((~val_df["exists"]).sum())
    missing_test = int((~test_df["exists"]).sum())
    if missing_val or missing_test:
        print(f"WARNING missing images: val={missing_val}, test={missing_test}")
        if missing_val:
            print(val_df.loc[~val_df["exists"], ["Path", "abs_path"]].head(5))
        if missing_test:
            print(test_df.loc[~test_df["exists"], ["Path", "abs_path"]].head(5))

    # Apply uncertain-label strategy and fill missing labels
    print("Applying uncertain policy...")
    val_df = apply_uncertain_policy(val_df, labels, policy)
    test_df = apply_uncertain_policy(test_df, labels, policy)

    # Split val_df into train and val_split_df
    print(f"Splitting VAL into train/val: train_frac={train_frac}")
    train_df, val_split_df = split_train_val_from_val(val_df, train_frac=train_frac, seed=seed)

    # Write processed CSVs
    train_csv = out_dir / "train_clean.csv"
    val_csv = out_dir / "val_clean.csv"
    test_csv = out_dir / "test_clean.csv"

    train_df.to_csv(train_csv, index=False)
    val_split_df.to_csv(val_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    # Save preprocessing report (useful for reproducibility / debugging)
    report = {
        "policy": policy,
        "seed": seed,
        "train_frac_from_val": train_frac,
        "counts": {
            "train_clean": len(train_df),
            "val_clean": len(val_split_df),
            "test_clean": len(test_df),
        },
        "missing_images": {"val": missing_val, "test": missing_test},
    }

    with open(out_dir / "prep_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print("\nSaved processed CSVs to:")
    print(" ", train_csv)
    print(" ", val_csv)
    print(" ", test_csv)
    print("\nCounts:", json.dumps(report["counts"], indent=2))
    return train_csv, val_csv, test_csv


# ============================================================
# C) Dataset + loaders
# ============================================================

class CheXpertDataset(Dataset):
    """
    PyTorch Dataset for multi-label classification.

    Expects CSV to contain:
      - abs_path : full path to image on disk
      - 14 label columns (CHEXPERT_LABELS_14)
    """
    def __init__(self, csv_path, labels=CHEXPERT_LABELS_14, transform=None):
        self.df = pd.read_csv(csv_path)
        self.labels = labels
        self.transform = transform

        if "abs_path" not in self.df.columns:
            raise ValueError("CSV must contain 'abs_path' column.")

        # Ensure labels are numeric floats (NaN -> 0)
        self.df[self.labels] = self.df[self.labels].apply(pd.to_numeric, errors="coerce").fillna(0.0)

        # Optional: warn if any image files are missing
        missing = (~self.df["abs_path"].apply(lambda p: Path(p).exists())).sum()
        if missing:
            print(f"WARNING: {missing} missing image paths in {csv_path}")

    def __len__(self):
        # Number of samples
        return len(self.df)

    def __getitem__(self, idx):
        # Fetch one row/sample
        row = self.df.iloc[idx]
        img_path = row["abs_path"]

        # Load image and force RGB (pretrained models expect 3 channels)
        img = Image.open(img_path).convert("RGB")

        # Multi-label target vector of shape (14,)
        y = torch.tensor(row[self.labels].to_numpy(dtype=np.float32))

        # Apply augmentations/transforms if provided
        if self.transform:
            img = self.transform(img)

        return img, y


def make_loaders(processed_dir: Path, batch_size=16, num_workers=0):
    """
    Create DataLoaders for train/val/test using standard ImageNet normalization.

    - train: includes light augmentation
    - eval : deterministic resize + normalize
    """
    processed_dir = Path(processed_dir)
    train_csv = processed_dir / "train_clean.csv"
    val_csv = processed_dir / "val_clean.csv"
    test_csv = processed_dir / "test_clean.csv"

    # Training transforms: resize + augmentation + normalize
    train_tfms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # Evaluation transforms: resize + normalize only (no randomization)
    eval_tfms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # Build datasets
    train_ds = CheXpertDataset(train_csv, transform=train_tfms)
    val_ds = CheXpertDataset(val_csv, transform=eval_tfms)
    test_ds = CheXpertDataset(test_csv, transform=eval_tfms)

    # DataLoaders: shuffle only for training
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader


def smoke_test(processed_dir: Path):
    """
    Quick sanity check:
    - Loads one batch from each loader
    - Prints tensor shapes
    """
    train_loader, val_loader, test_loader = make_loaders(processed_dir, batch_size=8, num_workers=0)
    x, y = next(iter(train_loader))
    print("Train batch x:", x.shape, "y:", y.shape)
    x2, y2 = next(iter(val_loader))
    print("Val batch x:", x2.shape, "y:", y2.shape)
    x3, y3 = next(iter(test_loader))
    print("Test batch x:", x3.shape, "y:", y3.shape)
    print("Smoke test OK.")


# ============================================================
# E) Training (ResNet34 + EfficientNet-B4)
# ============================================================

def build_resnet34(num_labels=14):
    """
    Load pretrained ResNet34 and replace final FC layer with (num_labels) outputs.
    """
    m = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
    m.fc = nn.Linear(m.fc.in_features, num_labels)
    return m


def build_effb4(num_labels=14):
    """
    Load pretrained EfficientNet-B4 and replace final classifier layer with (num_labels) outputs.
    """
    m = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.DEFAULT)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_labels)
    return m


@torch.no_grad()
def evaluate(model, loader, device):
    """
    Evaluate model on a loader.

    Metrics:
      - BCEWithLogitsLoss
      - micro_acc: average accuracy over ALL label decisions (multi-label)
      - pred_pos_rate: average predicted positive rate per label (diagnostic)
    """
    model.eval()
    crit = nn.BCEWithLogitsLoss()
    total_loss, n = 0.0, 0
    all_probs, all_y = [], []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        # Forward pass -> logits
        logits = model(x)

        # Multi-label loss (sigmoid is inside BCEWithLogitsLoss)
        loss = crit(logits, y)

        # Accumulate weighted by batch size to compute average loss later
        bs = x.size(0)
        total_loss += loss.item() * bs
        n += bs

        # Convert logits to probabilities in [0,1]
        probs = torch.sigmoid(logits).detach().cpu()
        all_probs.append(probs)
        all_y.append(y.detach().cpu())

    avg_loss = total_loss / max(n, 1)

    # Concatenate all batches
    probs = torch.cat(all_probs, dim=0)
    y_true = torch.cat(all_y, dim=0)

    # Threshold at 0.5 to get binary predictions per label
    preds = (probs >= 0.5).float()

    # Micro-accuracy across all labels and samples:
    # (pred == true) averaged over (N * 14) elements.
    micro_acc = (preds == y_true).float().mean().item()

    # Useful diagnostic: how often does the model predict "positive" for each label?
    pred_pos_rate = preds.mean(dim=0).tolist()

    return {
        "loss": float(avg_loss),
        "micro_acc": float(micro_acc),
        "pred_pos_rate": [float(v) for v in pred_pos_rate]
    }


def train_model(model, train_loader, val_loader, device, epochs=8, lr=3e-4):
    """
    Train model with AdamW + BCEWithLogitsLoss.
    Keeps the best model checkpoint based on val_loss.
    """
    model = model.to(device)
    crit = nn.BCEWithLogitsLoss()
    opt = optim.AdamW(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    best_state = None

    for ep in range(1, epochs + 1):
        model.train()
        running, n = 0.0, 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            # Clear gradients (set_to_none is slightly faster)
            opt.zero_grad(set_to_none=True)

            # Forward -> logits -> loss
            logits = model(x)
            loss = crit(logits, y)

            # Backprop + optimizer step
            loss.backward()
            opt.step()

            # Accumulate train loss
            bs = x.size(0)
            running += loss.item() * bs
            n += bs

        train_loss = running / max(n, 1)

        # Validation pass
        val_metrics = evaluate(model, val_loader, device)

        print(
            f"Epoch {ep:02d} | train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | val_micro_acc={val_metrics['micro_acc']:.4f}"
        )

        # Track best checkpoint by validation loss
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    return model


def train_two_models(processed_dir: Path, out_dir: Path, batch_size=16, num_workers=0, epochs=8, lr=3e-4):
    """
    Train and evaluate:
      - ResNet34
      - EfficientNet-B4

    Saves:
      - weights .pt
      - results JSON (val/test metrics + config)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    train_loader, val_loader, test_loader = make_loaders(
        processed_dir, batch_size=batch_size, num_workers=num_workers
    )

    # --- ResNet34 ---
    print("\n=== Training ResNet34 ===")
    resnet = build_resnet34(num_labels=len(CHEXPERT_LABELS_14))
    resnet = train_model(resnet, train_loader, val_loader, device, epochs=epochs, lr=lr)
    resnet_val = evaluate(resnet, val_loader, device)
    resnet_test = evaluate(resnet, test_loader, device)

    # Save learned weights
    resnet_path = out_dir / "resnet34_chexlocalize.pt"
    torch.save(resnet.state_dict(), resnet_path)

    print("[ResNet34] val:", resnet_val)
    print("[ResNet34] test:", resnet_test)

    # --- EfficientNet-B4 ---
    print("\n=== Training EfficientNet-B4 ===")
    effb4 = build_effb4(num_labels=len(CHEXPERT_LABELS_14))
    effb4 = train_model(effb4, train_loader, val_loader, device, epochs=epochs, lr=lr)
    eff_val = evaluate(effb4, val_loader, device)
    eff_test = evaluate(effb4, test_loader, device)

    # Save learned weights
    eff_path = out_dir / "effb4_chexlocalize.pt"
    torch.save(effb4.state_dict(), eff_path)

    print("[EffB4] val:", eff_val)
    print("[EffB4] test:", eff_test)

    # Save a single results JSON with config + metrics
    results = {
        "processed_dir": str(processed_dir),
        "device": device,
        "config": {"batch_size": batch_size, "epochs": epochs, "lr": lr, "num_workers": num_workers},
        "labels": CHEXPERT_LABELS_14,
        "resnet34": {"weights": str(resnet_path), "val": resnet_val, "test": resnet_test},
        "efficientnet_b4": {"weights": str(eff_path), "val": eff_val, "test": eff_test},
    }

    results_path = out_dir / "results_chexlocalize.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\nSaved weights + results:")
    print(" ", resnet_path)
    print(" ", eff_path)
    print(" ", results_path)


# ============================================================
# F) CLI main
# ============================================================

def main():
    """
    Command line interface entry point.

    Modes:
      - download_chexpert: download Stanford CheXpert small zip (optional)
      - prep: process CheXlocalize val/test into train/val/test CSVs
      - smoke: print batch shapes to verify loaders
      - train: train two models assuming processed CSVs exist
      - all: run prep then train
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["prep", "smoke", "train", "all", "download_chexpert"], default="prep")
    parser.add_argument("--policy", choices=["uzeros", "uones"], default="uzeros")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--train_frac", type=float, default=0.80)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()

    # Project root = folder that contains this script
    project_root = Path(__file__).resolve().parent

    # EDIT THIS to your real chexlocalize root:
    CHEXLOCALIZE_ROOT = Path(r"C:\Users\Ray\Desktop\downloads\chexlocalize\chexlocalize\CheXpert")

    # Where to write processed CSVs
    processed_out = project_root / "data" / "processed_chexlocalize"

    # Where to save weights/results (here: next to script)
    out_dir = project_root

    # Optional dataset download
    if args.mode == "download_chexpert":
        ensure_chexpert_downloaded_and_extracted(project_root)
        return

    # Preprocessing stage
    if args.mode in ["prep", "all"]:
        prep_chexlocalize(
            chexlocalize_root=CHEXLOCALIZE_ROOT,
            out_dir=processed_out,
            policy=args.policy,
            seed=args.seed,
            train_frac=args.train_frac,
        )

    # Loader sanity check stage
    if args.mode == "smoke":
        if not (processed_out / "train_clean.csv").exists():
            print("Processed CSVs not found. Run --mode prep first (or --mode all).")
            return
        smoke_test(processed_out)
        return

    # Training stage
    if args.mode in ["train", "all"]:
        if not (processed_out / "train_clean.csv").exists():
            print("Processed CSVs not found. Run --mode prep first (or --mode all).")
            return

        train_two_models(
            processed_dir=processed_out,
            out_dir=out_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            epochs=args.epochs,
            lr=args.lr,
        )
        return


if __name__ == "__main__":
    main()
