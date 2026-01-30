from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image

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

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row["abs_path"]
        img = Image.open(img_path).convert("RGB")

        y = torch.tensor(row[self.labels].values, dtype=torch.float32)

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
