from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

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


# Defining how big the dataset is, how to retrieve a sample at i-th index, and constructor for the datastruture. This is a pytorch dataset subclass
class CheXpertDataset(Dataset):
    def __init__(self, csv_path, labels=CHEXPERT_LABELS_14, transform=None):
        self.df = pd.read_csv(csv_path)
        self.labels = labels
        self.transform = transform

        self.df[self.labels] = self.df[self.labels].apply(
            pd.to_numeric, errors="coerce"
        )

        self.df[self.labels] = self.df[self.labels].fillna(0.0)
        self.df[self.labels] = self.df[self.labels].replace(-1.0, 0.0)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row["abs_path"]
        img = Image.open(img_path).convert("RGB")

        y = torch.from_numpy(row[self.labels].to_numpy(dtype=np.float32))
        if self.transform:
            img = self.transform(img)

        return img, y


from torchvision import transforms


# transforms here includes, resizing to 224x224, which is standard, and perform some augmentation such as flipping left right, and convert to a 3x224x224 tensor shape
train_tfms = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

val_tfms = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

from torch.utils.data import DataLoader

# Load dataset, starts from here and test your model with both the loader variables below
train_ds = CheXpertDataset(
    "data/processed_chexpert/train_clean.csv", transform=train_tfms
)
val_ds = CheXpertDataset("data/processed_chexpert/val_clean.csv", transform=val_tfms)

train_loader = DataLoader(
    train_ds, batch_size=32, shuffle=True, num_workers=4, pin_memory=True
)
val_loader = DataLoader(
    val_ds, batch_size=32, shuffle=False, num_workers=4, pin_memory=True
)


# ConvNext Model

import torch.nn as nn
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights

device = "cuda" if torch.cuda.is_available() else "cpu"


num_classes = 14
weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1

model = convnext_tiny(weights=weights)
in_features = model.classifier[2].in_features
model.classifier[2] = nn.Linear(in_features, num_classes)
model = model.to(device)

n_params = sum(p.numel() for p in model.parameters())
print(f"Params: {n_params / 1e6:.2f}M")

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

from tqdm import tqdm


def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    run_loss = 0.0

    for images, labels in tqdm(loader, leave=False):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)

        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        run_loss += loss.item() * images.size(0)
    return run_loss / len(loader.dataset)


def evaluate(model, loader, criterion, threshhold=0.5):
    model.eval()
    run_loss = 0.0
    tp = fp = fn = 0.0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device).float()
        logits = model(images)
        loss = criterion(logits, labels)
        run_loss += loss.item() * images.size(0)

        probs = torch.sigmoid(logits)
        preds = (probs >= threshhold).float()

        tp += (preds * labels).sum().item()
        fp += (preds * (1 - labels)).sum().item()
        fn += ((1 - preds) * labels).sum().item()

    avg_loss = run_loss / len(loader.dataset)
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    return avg_loss, f1


epochs = 10

best_val = float("inf")

for epoch in range(epochs):
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
    val_loss, val_acc = evaluate(model, val_loader, criterion)

    scheduler.step()

    print(
        f"Epoch {epoch + 1}/{epochs} | train_loss:{train_loss:.4f} | val_loss = {val_loss:.4f} | val_acc={val_acc:.4f}"
    )

    if val_loss < best_val:
        best_val = val_loss
        torch.save(model.state_dict(), "best_convnext.pt")
