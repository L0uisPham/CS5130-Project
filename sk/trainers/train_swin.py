import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets.chexpert import CheXpertDataset
from sk.model_wrappers.swin import SwinModel
from sk.trainers.engine import train_one_epoch, validate


def main():
    """
    Train and validate a Swin model on the CheXpert dataset,
    then evaluate the final saved model on the test_strat set.
    """

    # -------------------------
    # Training configuration
    # -------------------------
    num_epochs = 10
    freeze_epochs = 3
    batch_size = 32
    lr_head = 3e-4
    lr_full = 1e-5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------
    # Data transformations
    # -------------------------
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

    # -------------------------
    # Datasets
    # -------------------------
    train_dataset = CheXpertDataset(
        "data/CheXpert-v1.0-small/train_strat.csv",
        "data",
        train_transform
    )
    val_dataset = CheXpertDataset(
        "data/CheXpert-v1.0-small/valid_strat.csv",
        "data",
        val_transform
    )
    test_dataset = CheXpertDataset(
        "data/CheXpert-v1.0-small/test_strat.csv",
        "data",
        val_transform
    )

    # -------------------------
    # DataLoaders
    # -------------------------
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=4, pin_memory=True)

    # -------------------------
    # Model initialization
    # -------------------------
    model = SwinModel(
        model_name="swin_tiny_patch4_window7_224",
        num_classes=train_dataset.num_classes,
        pretrained=True
    ).to(device)

    # -------------------------
    # Loss function
    # -------------------------
    label_matrix = train_dataset.df[train_dataset.LABELS].values
    pos_weight = torch.tensor(
        (label_matrix.shape[0] - label_matrix.sum(axis=0)) /
        (label_matrix.sum(axis=0) + 1e-6),
        dtype=torch.float32,
        device=device
    )
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # -------------------------
    # Optimizer (head-only at first)
    # -------------------------
    model.freeze_backbone()
    optimizer = torch.optim.AdamW(model.get_trainable_params(), lr=lr_head)
    best_val_loss = float("inf")

    # -------------------------
    # Training loop
    # -------------------------
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        if epoch == freeze_epochs:
            model.unfreeze_backbone()
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr_full)

        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device)
        val_loss, aurocs = validate(
            model, val_loader, criterion, device, train_dataset.LABELS)

        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        for k, v in aurocs.items():
            if v is not None:
                print(f"  {k}: {v:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "sk/tuned_models/best_swin_model.pth")
            print("Model saved!")

    print("\nTraining complete")

    # -------------------------
    # Evaluate on TEST set
    # -------------------------
    print("\nEvaluating on test_strat set...")

    # Load the best model
    model.load_state_dict(torch.load(
        "sk/tuned_models/best_swin_model.pth", map_location=device))
    model.to(device)
    model.eval()

    test_loss, test_aurocs = validate(
        model, test_loader, criterion, device, train_dataset.LABELS)

    print(f"\nTest Loss: {test_loss:.4f}")
    print("Test AUROC scores per pathology:")
    for k, v in test_aurocs.items():
        if v is not None:
            print(f"  {k}: {v:.4f}")
    
    model_filename = f"sk/tuned_models/best_swin_model_{test_loss:.4f}.pth"
    torch.save(model.state_dict(), model_filename)
    print(f"Model saved as {model_filename}!")


if __name__ == "__main__":
    main()
