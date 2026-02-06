import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from datasets.chexpert import CheXpertDataset
from models.deit import DeiTModel
from engine import train_one_epoch, validate


def main():
    """
    Main function to train and validate the DeiT model on CheXpert.

    Includes data preprocessing, model initialization, loss and optimizer setup,
    optional backbone freezing, and saving the best model based on validation loss.
    """

    num_epochs = 10
    freeze_epochs = 3
    batch_size = 32
    lr_head = 3e-4
    lr_full = 1e-5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(5),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    train_dataset = CheXpertDataset(
        "data/CheXpert-v1.0-small/train_strat.csv", "data", train_transform
    )
    val_dataset = CheXpertDataset(
        "data/CheXpert-v1.0-small/valid_strat.csv", "data", val_transform
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size,
        shuffle=False, num_workers=4, pin_memory=True
    )

    model = DeiTModel(
        model_name="deit_small_patch16_224",
        num_classes=train_dataset.num_classes,
        pretrained=True
    ).to(device)

    label_matrix = train_dataset.df[train_dataset.LABELS].values
    pos_weight = torch.tensor(
        (label_matrix.shape[0] - label_matrix.sum(axis=0)) /
        (label_matrix.sum(axis=0) + 1e-6),
        dtype=torch.float32,
        device=device
    )

    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    model.freeze_backbone()
    optimizer = torch.optim.AdamW(
        model.get_trainable_params(), lr=lr_head
    )

    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        if epoch == freeze_epochs:
            model.unfreeze_backbone()
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=lr_full
            )

        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, aurocs = validate(
            model, val_loader, criterion, device, train_dataset.LABELS
        )

        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        for k, v in aurocs.items():
            if v is not None:
                print(f"  {k}: {v:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "models/best_model.pth")
            print("✓ Model saved")

    print("\nTraining complete")


if __name__ == "__main__":
    main()
