import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from sk.dataset.chexpert import CheXpertDataset
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

"""
OUTPUT

Epoch 1/10
Train Loss: 1.0706 | Val Loss: 1.1009
  Atelectasis: 0.6022
  Cardiomegaly: 0.6735
  Consolidation: 0.6082
  Edema: 0.8022
  Enlarged Cardiomediastinum: 0.4505
  Fracture: 0.7028
  Lung Lesion: 0.7549
  Lung Opacity: 0.6109
  Pleural Effusion: 0.7843
  Pleural Other: 0.7378
  Pneumonia: 0.6148
  Pneumothorax: 0.7988
  Support Devices: 0.7929
Model saved!

Epoch 2/10
Train Loss: 1.0533 | Val Loss: 1.0824
  Atelectasis: 0.6541
  Cardiomegaly: 0.6725
  Consolidation: 0.6498
  Edema: 0.8047
  Enlarged Cardiomediastinum: 0.6131
  Fracture: 0.7244
  Lung Lesion: 0.7464
  Lung Opacity: 0.6124
  Pleural Effusion: 0.7900
  Pleural Other: 0.7566
  Pneumonia: 0.6160
  Pneumothorax: 0.8042
  Support Devices: 0.8002
Model saved!

Epoch 3/10
Train Loss: 1.0491 | Val Loss: 1.0864
  Atelectasis: 0.6359
  Cardiomegaly: 0.6518
  Consolidation: 0.6881
  Edema: 0.8026
  Enlarged Cardiomediastinum: 0.6792
  Fracture: 0.7860
  Lung Lesion: 0.7344
  Lung Opacity: 0.6111
  Pleural Effusion: 0.7869
  Pleural Other: 0.7522
  Pneumonia: 0.5768
  Pneumothorax: 0.8105
  Support Devices: 0.8055

Epoch 4/10
Train Loss: 1.0031 | Val Loss: 1.0799
  Atelectasis: 0.6972
  Cardiomegaly: 0.8000
  Consolidation: 0.6346
  Edema: 0.8398
  Enlarged Cardiomediastinum: 0.4908
  Fracture: 0.6788
  Lung Lesion: 0.6951
  Lung Opacity: 0.6672
  Pleural Effusion: 0.8278
  Pleural Other: 0.7843
  Pneumonia: 0.6580
  Pneumothorax: 0.9161
  Support Devices: 0.8188
Model saved!

Epoch 5/10
Train Loss: 0.9595 | Val Loss: 1.0510
  Atelectasis: 0.7186
  Cardiomegaly: 0.7928
  Consolidation: 0.6217
  Edema: 0.8492
  Enlarged Cardiomediastinum: 0.6082
  Fracture: 0.7766
  Lung Lesion: 0.6987
  Lung Opacity: 0.6829
  Pleural Effusion: 0.8481
  Pleural Other: 0.7909
  Pneumonia: 0.6425
  Pneumothorax: 0.9217
  Support Devices: 0.8170
Model saved!

Epoch 6/10
Train Loss: 0.9356 | Val Loss: 1.0070
  Atelectasis: 0.7308
  Cardiomegaly: 0.8244
  Consolidation: 0.6301
  Edema: 0.8466
  Enlarged Cardiomediastinum: 0.6519
  Fracture: 0.7770
  Lung Lesion: 0.6879
  Lung Opacity: 0.7097
  Pleural Effusion: 0.8713
  Pleural Other: 0.7948
  Pneumonia: 0.6539
  Pneumothorax: 0.9061
  Support Devices: 0.8224
Model saved!

Epoch 7/10
Train Loss: 0.9173 | Val Loss: 1.0221
  Atelectasis: 0.7077
  Cardiomegaly: 0.7998
  Consolidation: 0.5916
  Edema: 0.8439
  Enlarged Cardiomediastinum: 0.6123
  Fracture: 0.8272
  Lung Lesion: 0.7116
  Lung Opacity: 0.7149
  Pleural Effusion: 0.8633
  Pleural Other: 0.7871
  Pneumonia: 0.6445
  Pneumothorax: 0.9058
  Support Devices: 0.8220

Epoch 8/10
Train Loss: 0.9012 | Val Loss: 1.0968
  Atelectasis: 0.7189
  Cardiomegaly: 0.7960
  Consolidation: 0.5879
  Edema: 0.8524
  Enlarged Cardiomediastinum: 0.6514
  Fracture: 0.7766
  Lung Lesion: 0.7241
  Lung Opacity: 0.7165
  Pleural Effusion: 0.8551
  Pleural Other: 0.7671
  Pneumonia: 0.6388
  Pneumothorax: 0.9086
  Support Devices: 0.8341

Epoch 9/10
Train Loss: 0.8881 | Val Loss: 1.0754
  Atelectasis: 0.7174
  Cardiomegaly: 0.8235
  Consolidation: 0.5811
  Edema: 0.8536
  Enlarged Cardiomediastinum: 0.6368
  Fracture: 0.7937
  Lung Lesion: 0.7375
  Lung Opacity: 0.7136
  Pleural Effusion: 0.8704
  Pleural Other: 0.7588
  Pneumonia: 0.6311
  Pneumothorax: 0.9070
  Support Devices: 0.8217

Epoch 10/10
Train Loss: 0.8750 | Val Loss: 1.0803
  Atelectasis: 0.7230
  Cardiomegaly: 0.8160
  Consolidation: 0.6037
  Edema: 0.8446
  Enlarged Cardiomediastinum: 0.6351
  Fracture: 0.7631
  Lung Lesion: 0.7129
  Lung Opacity: 0.7187
  Pleural Effusion: 0.8748
  Pleural Other: 0.7472
  Pneumonia: 0.6466
  Pneumothorax: 0.9217
  Support Devices: 0.8249

Training complete

Evaluating on test_strat set...

Test Loss: 1.0060
Test AUROC scores per pathology:
  Atelectasis: 0.6056
  Cardiomegaly: 0.8074
  Consolidation: 0.7242
  Edema: 0.7900
  Enlarged Cardiomediastinum: 0.4782
  Fracture: 0.8930
  Lung Lesion: 0.6958
  Lung Opacity: 0.7261
  Pleural Effusion: 0.8547
  Pleural Other: 0.7826
  Pneumonia: 0.7160
  Pneumothorax: 0.8902
  Support Devices: 0.8303
Model saved as sk/tuned_models/best_swin_model_1.0060.pth!
"""