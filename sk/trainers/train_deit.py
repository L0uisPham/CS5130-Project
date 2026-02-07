import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from sk.dataset.chexpert import CheXpertDataset
from sk.model_wrappers.deit import DeiTModel
from sk.trainers.engine import train_one_epoch, validate


def main():
    """
    Train and validate a DeiT model on the CheXpert dataset,
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
    model = DeiTModel(
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
            torch.save(model.state_dict(), "sk/tuned_models/best_deit_model.pth")
            print("Model saved!")

    print("\nTraining complete")

    # -------------------------
    # Evaluate on TEST set
    # -------------------------
    print("\nEvaluating on test_strat set...")

    # Load the best model
    model.load_state_dict(torch.load(
        "sk/tuned_models/best_deit_model.pth", map_location=device))
    model.to(device)
    model.eval()

    test_loss, test_aurocs = validate(
        model, test_loader, criterion, device, train_dataset.LABELS)

    print(f"\nTest Loss: {test_loss:.4f}")
    print("Test AUROC scores per pathology:")
    for k, v in test_aurocs.items():
        if v is not None:
            print(f"  {k}: {v:.4f}")

    model_filename = f"sk/tuned_models/best_deit_model_{test_loss:.4f}.pth"
    torch.save(model.state_dict(), model_filename)
    print(f"Model saved as {model_filename}!")


if __name__ == "__main__":
    main()

"""
OUTPUT

Epoch 1/10
Train Loss: 1.0717 | Val Loss: 1.1276
  Atelectasis: 0.6669
  Cardiomegaly: 0.6276
  Consolidation: 0.6276
  Edema: 0.7898
  Enlarged Cardiomediastinum: 0.5234
  Fracture: 0.5646
  Lung Lesion: 0.7987
  Lung Opacity: 0.6201
  Pleural Effusion: 0.7432
  Pleural Other: 0.7140
  Pneumonia: 0.5728
  Pneumothorax: 0.7271
  Support Devices: 0.8124
Model saved!

Epoch 2/10
Train Loss: 1.0553 | Val Loss: 1.1264
  Atelectasis: 0.6636
  Cardiomegaly: 0.6569
  Consolidation: 0.6329
  Edema: 0.7945
  Enlarged Cardiomediastinum: 0.5165
  Fracture: 0.6584
  Lung Lesion: 0.7737
  Lung Opacity: 0.6149
  Pleural Effusion: 0.7523
  Pleural Other: 0.7456
  Pneumonia: 0.5870
  Pneumothorax: 0.7736
  Support Devices: 0.8161
Model saved!

Epoch 3/10
Train Loss: 1.0515 | Val Loss: 1.0920
  Atelectasis: 0.6917
  Cardiomegaly: 0.6811
  Consolidation: 0.6416
  Edema: 0.7965
  Enlarged Cardiomediastinum: 0.5108
  Fracture: 0.6119
  Lung Lesion: 0.7571
  Lung Opacity: 0.6031
  Pleural Effusion: 0.7492
  Pleural Other: 0.7594
  Pneumonia: 0.6396
  Pneumothorax: 0.7710
  Support Devices: 0.8155
Model saved!

Epoch 4/10
Train Loss: 0.9977 | Val Loss: 1.0623
  Atelectasis: 0.7251
  Cardiomegaly: 0.8188
  Consolidation: 0.6439
  Edema: 0.8356
  Enlarged Cardiomediastinum: 0.5654
  Fracture: 0.7460
  Lung Lesion: 0.7371
  Lung Opacity: 0.6737
  Pleural Effusion: 0.8562
  Pleural Other: 0.7627
  Pneumonia: 0.6523
  Pneumothorax: 0.9273
  Support Devices: 0.8089
Model saved!

Epoch 5/10
Train Loss: 0.9523 | Val Loss: 1.0562
  Atelectasis: 0.7228
  Cardiomegaly: 0.8061
  Consolidation: 0.6400
  Edema: 0.8376
  Enlarged Cardiomediastinum: 0.6103
  Fracture: 0.8015
  Lung Lesion: 0.7348
  Lung Opacity: 0.6752
  Pleural Effusion: 0.8535
  Pleural Other: 0.7306
  Pneumonia: 0.6714
  Pneumothorax: 0.9285
  Support Devices: 0.8223
Model saved!

Epoch 6/10
Train Loss: 0.9270 | Val Loss: 1.0232
  Atelectasis: 0.7098
  Cardiomegaly: 0.8343
  Consolidation: 0.6364
  Edema: 0.8358
  Enlarged Cardiomediastinum: 0.6388
  Fracture: 0.8113
  Lung Lesion: 0.7049
  Lung Opacity: 0.6841
  Pleural Effusion: 0.8832
  Pleural Other: 0.7566
  Pneumonia: 0.6498
  Pneumothorax: 0.8967
  Support Devices: 0.8166
Model saved!

Epoch 7/10
Train Loss: 0.9076 | Val Loss: 0.9940
  Atelectasis: 0.7167
  Cardiomegaly: 0.8570
  Consolidation: 0.6138
  Edema: 0.8440
  Enlarged Cardiomediastinum: 0.6025
  Fracture: 0.8100
  Lung Lesion: 0.7250
  Lung Opacity: 0.6951
  Pleural Effusion: 0.8829
  Pleural Other: 0.8042
  Pneumonia: 0.7020
  Pneumothorax: 0.9353
  Support Devices: 0.8134
Model saved!

Epoch 8/10
Train Loss: 0.8897 | Val Loss: 1.0697
  Atelectasis: 0.7073
  Cardiomegaly: 0.8315
  Consolidation: 0.6248
  Edema: 0.8383
  Enlarged Cardiomediastinum: 0.6217
  Fracture: 0.8267
  Lung Lesion: 0.7121
  Lung Opacity: 0.6975
  Pleural Effusion: 0.8837
  Pleural Other: 0.7329
  Pneumonia: 0.6543
  Pneumothorax: 0.9393
  Support Devices: 0.7990

Epoch 9/10
Train Loss: 0.8716 | Val Loss: 1.0079
  Atelectasis: 0.6982
  Cardiomegaly: 0.8365
  Consolidation: 0.5631
  Edema: 0.8178
  Enlarged Cardiomediastinum: 0.6413
  Fracture: 0.8594
  Lung Lesion: 0.7482
  Lung Opacity: 0.6982
  Pleural Effusion: 0.8778
  Pleural Other: 0.8208
  Pneumonia: 0.6796
  Pneumothorax: 0.9271
  Support Devices: 0.8257

Epoch 10/10
Train Loss: 0.8526 | Val Loss: 1.0266
  Atelectasis: 0.6971
  Cardiomegaly: 0.8412
  Consolidation: 0.6280
  Edema: 0.8365
  Enlarged Cardiomediastinum: 0.6947
  Fracture: 0.8569
  Lung Lesion: 0.7554
  Lung Opacity: 0.7038
  Pleural Effusion: 0.8828
  Pleural Other: 0.7218
  Pneumonia: 0.6315
  Pneumothorax: 0.9299
  Support Devices: 0.8216

Training complete

Evaluating on test_strat set...

Test Loss: 1.0176
Test AUROC scores per pathology:
  Atelectasis: 0.6130
  Cardiomegaly: 0.7994
  Consolidation: 0.7308
  Edema: 0.7694
  Enlarged Cardiomediastinum: 0.4767
  Fracture: 0.8348
  Lung Lesion: 0.7383
  Lung Opacity: 0.7247
  Pleural Effusion: 0.8629
  Pleural Other: 0.7435
  Pneumonia: 0.6563
  Pneumothorax: 0.8887
  Support Devices: 0.8488
Model saved as sk/tuned_models/best_deit_model_1.0176.pth!
"""