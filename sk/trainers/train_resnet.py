import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from sk.datasets.chexpert import CheXpertDataset
from sk.model_wrappers.resnet import ResNetModel
from sk.trainers.engine import train_one_epoch, validate


def main():
    """
    Train and validate a resnet model on the CheXpert dataset,
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
    model = ResNetModel(
        model_name="resnet34",
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
            torch.save(model.state_dict(), "sk/tuned_models/best_resnet_model.pth")
            print("Model saved!")

    print("\nTraining complete")

    # -------------------------
    # Evaluate on TEST set
    # -------------------------
    print("\nEvaluating on test_strat set...")

    # Load the best model
    model.load_state_dict(torch.load(
        "sk/tuned_models/best_resnet_model.pth", map_location=device))
    model.to(device)
    model.eval()

    test_loss, test_aurocs = validate(
        model, test_loader, criterion, device, train_dataset.LABELS)

    print(f"\nTest Loss: {test_loss:.4f}")
    print("Test AUROC scores per pathology:")
    for k, v in test_aurocs.items():
        if v is not None:
            print(f"  {k}: {v:.4f}")

    model_filename = f"sk/tuned_models/best_resnet_model_{test_loss:.4f}.pth"
    torch.save(model.state_dict(), model_filename)
    print(f"Model saved as {model_filename}!")


if __name__ == "__main__":
    main()

"""
OUTPUT

Epoch 1/10
Train Loss: 1.1072 | Val Loss: 1.1701
  Atelectasis: 0.6149
  Cardiomegaly: 0.6405
  Consolidation: 0.5708
  Edema: 0.7856
  Enlarged Cardiomediastinum: 0.5247
  Fracture: 0.4452
  Lung Lesion: 0.7237
  Lung Opacity: 0.5934
  Pleural Effusion: 0.6698
  Pleural Other: 0.6737
  Pneumonia: 0.5903
  Pneumothorax: 0.6748
  Support Devices: 0.7527
Model saved!

Epoch 2/10
Train Loss: 1.0877 | Val Loss: 1.1612
  Atelectasis: 0.6152
  Cardiomegaly: 0.6518
  Consolidation: 0.5907
  Edema: 0.7921
  Enlarged Cardiomediastinum: 0.6213
  Fracture: 0.5169
  Lung Lesion: 0.7165
  Lung Opacity: 0.5964
  Pleural Effusion: 0.6688
  Pleural Other: 0.6897
  Pneumonia: 0.6115
  Pneumothorax: 0.6846
  Support Devices: 0.7536
Model saved!

Epoch 3/10
Train Loss: 1.0828 | Val Loss: 1.1450
  Atelectasis: 0.6205
  Cardiomegaly: 0.6574
  Consolidation: 0.5977
  Edema: 0.7950
  Enlarged Cardiomediastinum: 0.5503
  Fracture: 0.5263
  Lung Lesion: 0.7254
  Lung Opacity: 0.6003
  Pleural Effusion: 0.6609
  Pleural Other: 0.7251
  Pneumonia: 0.6103
  Pneumothorax: 0.7058
  Support Devices: 0.7560
Model saved!

Epoch 4/10
Train Loss: 1.0652 | Val Loss: 1.1338
  Atelectasis: 0.6246
  Cardiomegaly: 0.6441
  Consolidation: 0.6266
  Edema: 0.7949
  Enlarged Cardiomediastinum: 0.6066
  Fracture: 0.4868
  Lung Lesion: 0.7531
  Lung Opacity: 0.6128
  Pleural Effusion: 0.6927
  Pleural Other: 0.7030
  Pneumonia: 0.6747
  Pneumothorax: 0.7495
  Support Devices: 0.7582
Model saved!

Epoch 5/10
Train Loss: 1.0463 | Val Loss: 1.1195
  Atelectasis: 0.6389
  Cardiomegaly: 0.6860
  Consolidation: 0.6418
  Edema: 0.7780
  Enlarged Cardiomediastinum: 0.6608
  Fracture: 0.6074
  Lung Lesion: 0.7411
  Lung Opacity: 0.5980
  Pleural Effusion: 0.7307
  Pleural Other: 0.7268
  Pneumonia: 0.6625
  Pneumothorax: 0.7843
  Support Devices: 0.7628
Model saved!

Epoch 6/10
Train Loss: 1.0318 | Val Loss: 1.1120
  Atelectasis: 0.6198
  Cardiomegaly: 0.7031
  Consolidation: 0.6271
  Edema: 0.7844
  Enlarged Cardiomediastinum: 0.6025
  Fracture: 0.5695
  Lung Lesion: 0.7830
  Lung Opacity: 0.6385
  Pleural Effusion: 0.7698
  Pleural Other: 0.7611
  Pneumonia: 0.6380
  Pneumothorax: 0.7853
  Support Devices: 0.7660
Model saved!

Epoch 7/10
Train Loss: 1.0200 | Val Loss: 1.1220
  Atelectasis: 0.6395
  Cardiomegaly: 0.7152
  Consolidation: 0.6157
  Edema: 0.7926
  Enlarged Cardiomediastinum: 0.5858
  Fracture: 0.6066
  Lung Lesion: 0.7482
  Lung Opacity: 0.6301
  Pleural Effusion: 0.7896
  Pleural Other: 0.7329
  Pneumonia: 0.6355
  Pneumothorax: 0.8089
  Support Devices: 0.7728

Epoch 8/10
Train Loss: 1.0106 | Val Loss: 1.0984
  Atelectasis: 0.6619
  Cardiomegaly: 0.7440
  Consolidation: 0.6276
  Edema: 0.7998
  Enlarged Cardiomediastinum: 0.5732
  Fracture: 0.6302
  Lung Lesion: 0.7781
  Lung Opacity: 0.6361
  Pleural Effusion: 0.7983
  Pleural Other: 0.7517
  Pneumonia: 0.6196
  Pneumothorax: 0.8234
  Support Devices: 0.7960
Model saved!

Epoch 9/10
Train Loss: 1.0017 | Val Loss: 1.1076
  Atelectasis: 0.6501
  Cardiomegaly: 0.7218
  Consolidation: 0.6332
  Edema: 0.8024
  Enlarged Cardiomediastinum: 0.5919
  Fracture: 0.6604
  Lung Lesion: 0.7879
  Lung Opacity: 0.6393
  Pleural Effusion: 0.8044
  Pleural Other: 0.7445
  Pneumonia: 0.6343
  Pneumothorax: 0.8131
  Support Devices: 0.7851

Epoch 10/10
Train Loss: 0.9935 | Val Loss: 1.1219
  Atelectasis: 0.6580
  Cardiomegaly: 0.7474
  Consolidation: 0.6446
  Edema: 0.8087
  Enlarged Cardiomediastinum: 0.5532
  Fracture: 0.6290
  Lung Lesion: 0.7781
  Lung Opacity: 0.6231
  Pleural Effusion: 0.8048
  Pleural Other: 0.7306
  Pneumonia: 0.6245
  Pneumothorax: 0.8493
  Support Devices: 0.8011

Training complete

Evaluating on test_strat set...

Test Loss: 1.0876
Test AUROC scores per pathology:
  Atelectasis: 0.5958
  Cardiomegaly: 0.7549
  Consolidation: 0.6377
  Edema: 0.7814
  Enlarged Cardiomediastinum: 0.6273
  Fracture: 0.7173
  Lung Lesion: 0.6934
  Lung Opacity: 0.6705
  Pleural Effusion: 0.8055
  Pleural Other: 0.7348
  Pneumonia: 0.6331
  Pneumothorax: 0.7197
  Support Devices: 0.7641
Model saved as sk/tuned_models/best_resnet_model_1.0876.pth!
"""