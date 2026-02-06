import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image


class CheXpertDataset(Dataset):
    """
    PyTorch Dataset for the CheXpert dataset.

    Attributes:
        LABELS (list of str): List of 14 CheXpert pathology labels.
        df (pd.DataFrame): DataFrame containing image paths and labels.
        root_dir (str): Root directory where images are stored.
        transform (callable, optional): Transformations to apply to images.
    """

    LABELS = [
        "Atelectasis", "Cardiomegaly", "Consolidation",
        "Edema", "Enlarged Cardiomediastinum", "Fracture",
        "Lung Lesion", "Lung Opacity", "Pleural Effusion",
        "Pleural Other", "Pneumonia", "Pneumothorax",
        "Support Devices"
    ]

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Initializes the CheXpert dataset.

        Args:
            csv_file (str): Path to the CSV file containing image paths and labels.
            root_dir (str): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on an image.
        """

        self.df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

        # Ensure label columns are numeric and fill NaNs with 0.0
        self.df[self.LABELS] = (
            self.df[self.LABELS]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0.0)
            .astype("float32")
        )

    def __len__(self):
        """
        Returns:
            int: Total number of samples in the dataset.
        """

        return len(self.df)

    def __getitem__(self, idx):
        """
        Retrieves an image and its labels at a specific index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: (image, label) where image is a transformed PIL image (or tensor)
                   and label is a tensor of shape (num_classes,).
        """

        img_path = os.path.join(self.root_dir, self.df.iloc[idx]["Path"])
        image = Image.open(img_path).convert("RGB")

        label = torch.from_numpy(
            self.df.loc[self.df.index[idx], self.LABELS]
                .to_numpy(dtype="float32", copy=True)
        )

        if self.transform:
            image = self.transform(image)

        return image, label

    @property
    def num_classes(self):
        """
        Returns:
            int: Number of classes (labels) in the dataset.
        """

        return len(self.LABELS)
