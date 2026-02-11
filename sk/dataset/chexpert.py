import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torchvision import transforms


class CheXpert(Dataset):
    LABELS = [
<<<<<<< Updated upstream
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
        "sex_male",
        "sex_female",
        "age_0_29",
        "age_30_44",
        "age_45_59",
        "age_60_74",
        "age_75_plus"
=======
        "Atelectasis",
        "Cardiomegaly",
        "Consolidation",
        "Edema",
        "Enlarged Cardiomediastinum",
        "Fracture",
        "Lung Lesion",
        "Lung Opacity",
        "Pleural Effusion",
        "Pleural Other",
        "Pneumonia",
        "Pneumothorax",
        "Support Devices",
>>>>>>> Stashed changes
    ]

    ROOT_DIR = "data"

    TRAIN_STRAT_PATH = "data/CheXpert-v1.0-small/train_strat.csv"
    VALID_STRAT_PATH = "data/CheXpert-v1.0-small/valid_strat.csv"
    TEST_STRAT_PATH = "data/CheXpert-v1.0-small/test_strat.csv"

    TRAIN_TRANSFORM = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(5),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    VALID_TRANSFORM = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    def __init__(self, set):
        set_lower = set.strip().lower()
        self.set = set_lower

        if set_lower == "train":
            path = self.TRAIN_STRAT_PATH
            self.transform = self.TRAIN_TRANSFORM
            self.shuffle = True
        elif set_lower == "valid":
            path = self.VALID_STRAT_PATH
            self.transform = self.VALID_TRANSFORM
            self.shuffle = False
        elif set_lower == "test":
            path = self.TEST_STRAT_PATH
            self.transform = self.VALID_TRANSFORM
            self.shuffle = False
        else:
            raise ValueError(
                "set parameter can only be 'train', 'valid', or 'test'")

        self.df = pd.read_csv(path)
        self.df[self.LABELS] = self.df[self.LABELS].apply(
            pd.to_numeric, errors="coerce"
        ).fillna(0.0).astype("float32")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = os.path.join(self.ROOT_DIR, self.df.iloc[idx]["Path"])
        image = Image.open(img_path).convert("RGB")
<<<<<<< Updated upstream
        label = torch.tensor(
            self.df.loc[idx, self.LABELS].values, dtype=torch.float32)
        image = self.transform(image)
=======

        label = torch.from_numpy(
            self.df.loc[self.df.index[idx], self.LABELS].to_numpy(
                dtype="float32", copy=True
            )
        )

        if self.transform:
            image = self.transform(image)

>>>>>>> Stashed changes
        return image, label

    @property
    def num_classes(self):
        return len(self.LABELS)

    def get_loader(self, batch_size=32, num_workers=4, pin_memory=True):
        return DataLoader(self, batch_size=batch_size,
                          shuffle=self.shuffle,
                          num_workers=num_workers,
                          pin_memory=pin_memory)
