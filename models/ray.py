"""
train_models_from_processed_chexpert.py

Train ResNet34 + EfficientNet-B4 ONLY using the already-processed CheXpert splits:

  data/processed_chexpert/train_clean.csv
  data/processed_chexpert/val_clean.csv
  data/processed_chexpert/test_clean.csv
  (optional) data/processed_chexpert/valid_official.csv

This script DOES NOT:
- download datasets
- re-split patients
- modify CSVs
- run CheXlocalize preprocessing

USAGE (PowerShell):
  python .\scripts\train_models_from_processed_chexpert.py --mode smoke
  python .\scripts\train_models_from_processed_chexpert.py --mode train --epochs 2
"""

import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms, models


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


# ============================================================
# Dataset + loaders (ONLY reads existing processed CSVs)
# ============================================================

class CheXpertDataset(Dataset):
    def __init__(self, csv_path: Path, labels=CHEXPERT_LABELS_14, transform=None, max_rows=None):
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing CSV: {csv_path}")

        self.df = pd.read_csv(csv_path)
        if max_rows is not None:
            self.df = self.df.head(int(max_rows)).reset_index(drop=True)

        self.labels = labels
        self.transform = transform

        if "abs_path" not in self.df.columns:
            raise ValueError(f"{csv_path.name} must contain 'abs_path' column.")

        # Ensure labels numeric
        for c in self.labels:
            if c not in self.df.columns:
                raise ValueError(f"{csv_path.name} missing label column: {c}")
        self.df[self.labels] = self.df[self.labels].apply(pd.to_numeric, errors="coerce").fillna(0.0)

        # Warn about missing image files
        missing = (~self.df["abs_path"].apply(lambda p: Path(p).exists())).sum()
        if missing:
            print(f"WARNING: {missing} missing image paths referenced by {csv_path.name}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["abs_path"]).convert("RGB")
        y = torch.tensor(row[self.labels].to_numpy(dtype=np.float32))
        if self.transform:
            img = self.transform(img)
        return img, y


def make_loaders(processed_dir: Path, batch_size=16, num_workers=0, max_rows=None):
    processed_dir = Path(processed_dir)

    train_csv = processed_dir / "train_clean.csv"
    val_csv = processed_dir / "val_clean.csv"
    test_csv = processed_dir / "test_clean.csv"

    # Standard ImageNet preprocessing for pretrained models
    train_tfms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    eval_tfms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    train_ds = CheXpertDataset(train_csv, transform=train_tfms, max_rows=max_rows)
    val_ds = CheXpertDataset(val_csv, transform=eval_tfms, max_rows=max_rows)
    test_ds = CheXpertDataset(test_csv, transform=eval_tfms, max_rows=max_rows)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader


def smoke_test(processed_dir: Path, max_rows=None):
    tr, va, te = make_loaders(processed_dir, batch_size=8, num_workers=0, max_rows=max_rows)
    x, y = next(iter(tr)); print("Train batch:", x.shape, y.shape)
    x, y = next(iter(va)); print("Val batch:", x.shape, y.shape)
    x, y = next(iter(te)); print("Test batch:", x.shape, y.shape)
    print("Smoke test OK.")


# ============================================================
# Models + train/eval
# ============================================================

def build_resnet34(num_labels=14):
    m = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
    m.fc = nn.Linear(m.fc.in_features, num_labels)
    return m


def build_effb4(num_labels=14):
    m = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.DEFAULT)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_labels)
    return m


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    crit = nn.BCEWithLogitsLoss()
    total_loss, n = 0.0, 0
    all_probs, all_y = [], []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = crit(logits, y)

        bs = x.size(0)
        total_loss += loss.item() * bs
        n += bs

        probs = torch.sigmoid(logits).detach().cpu()
        all_probs.append(probs)
        all_y.append(y.detach().cpu())

    avg_loss = total_loss / max(n, 1)
    probs = torch.cat(all_probs, dim=0)
    y_true = torch.cat(all_y, dim=0)

    preds = (probs >= 0.5).float()
    micro_acc = (preds == y_true).float().mean().item()

    return {"loss": float(avg_loss), "micro_acc": float(micro_acc)}


def train_model(model, train_loader, val_loader, device, epochs=8, lr=3e-4):
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

            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = crit(logits, y)
            loss.backward()
            opt.step()

            bs = x.size(0)
            running += loss.item() * bs
            n += bs

        train_loss = running / max(n, 1)
        val_metrics = evaluate(model, val_loader, device)

        print(f"Epoch {ep:02d} | train_loss={train_loss:.4f} | "
              f"val_loss={val_metrics['loss']:.4f} | val_micro_acc={val_metrics['micro_acc']:.4f}")

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def train_two_models(processed_dir: Path, out_dir: Path, batch_size=16, num_workers=0, epochs=8, lr=3e-4, max_rows=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    train_loader, val_loader, test_loader = make_loaders(
        processed_dir, batch_size=batch_size, num_workers=num_workers, max_rows=max_rows
    )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ResNet34
    print("\n=== Training ResNet34 ===")
    resnet = build_resnet34(num_labels=len(CHEXPERT_LABELS_14))
    resnet = train_model(resnet, train_loader, val_loader, device, epochs=epochs, lr=lr)
    res_val = evaluate(resnet, val_loader, device)
    res_test = evaluate(resnet, test_loader, device)
    res_path = out_dir / "resnet34_chexpert.pt"
    torch.save(resnet.state_dict(), res_path)
    print("[ResNet34] val:", res_val)
    print("[ResNet34] test:", res_test)

    # EfficientNet-B4
    print("\n=== Training EfficientNet-B4 ===")
    eff = build_effb4(num_labels=len(CHEXPERT_LABELS_14))
    eff = train_model(eff, train_loader, val_loader, device, epochs=epochs, lr=lr)
    eff_val = evaluate(eff, val_loader, device)
    eff_test = evaluate(eff, test_loader, device)
    eff_path = out_dir / "effb4_chexpert.pt"
    torch.save(eff.state_dict(), eff_path)
    print("[EffB4] val:", eff_val)
    print("[EffB4] test:", eff_test)

    results = {
        "processed_dir": str(Path(processed_dir).resolve()),
        "device": device,
        "config": {"batch_size": batch_size, "epochs": epochs, "lr": lr, "num_workers": num_workers, "max_rows": max_rows},
        "labels": CHEXPERT_LABELS_14,
        "resnet34": {"weights": str(res_path), "val": res_val, "test": res_test},
        "efficientnet_b4": {"weights": str(eff_path), "val": eff_val, "test": eff_test},
    }

    results_path = out_dir / "results_chexpert.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\nSaved:")
    print(" ", res_path)
    print(" ", eff_path)
    print(" ", results_path)


# ============================================================
# CLI
# ============================================================

import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "train"], default="smoke")
    parser.add_argument("--processed_dir", type=str, default="data/processed_chexpert")
    parser.add_argument("--out_dir", type=str, default="models")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_rows", type=int, default=None, help="Optional row limit for quick runs")
    args = parser.parse_args()

    processed_dir = Path(args.processed_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Hard fail if the split files aren’t present
    for f in ["train_clean.csv", "val_clean.csv", "test_clean.csv"]:
        if not (processed_dir / f).exists():
            raise FileNotFoundError(f"Expected split file not found: {processed_dir / f}")

    if args.mode == "smoke":
        smoke_test(processed_dir, max_rows=args.max_rows)
        return

    if args.mode == "train":
        train_two_models(
            processed_dir=processed_dir,
            out_dir=out_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            epochs=args.epochs,
            lr=args.lr,
            max_rows=args.max_rows,
        )
        return

if __name__ == "__main__":
    main()
