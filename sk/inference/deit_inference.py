import random
import torch
from torchvision import transforms
from sk.dataset.chexpert import CheXpertDataset
from sk.model_wrappers.deit import DeiTModel


def main():
    # -------------------------
    # Configuration
    # -------------------------
    model_path = "sk/tuned_models/best_deit_model.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------
    # Transforms (same as val/test)
    # -------------------------
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        ),
    ])

    # -------------------------
    # Dataset
    # -------------------------
    test_dataset = CheXpertDataset(
        "data/CheXpert-v1.0-small/test_strat.csv",
        "data",
        test_transform
    )

    # -------------------------
    # Random sample selection
    # -------------------------
    idx = random.randint(0, len(test_dataset) - 1)
    image, label = test_dataset[idx]

    image = image.unsqueeze(0).to(device)  # add batch dimension

    # -------------------------
    # Model
    # -------------------------
    model = DeiTModel(
        model_name="deit_small_patch16_224",
        num_classes=test_dataset.num_classes,
        pretrained=False
    )

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # -------------------------
    # Inference
    # -------------------------
    with torch.no_grad():
        logits = model(image)
        probs = torch.sigmoid(logits).squeeze(0).cpu()

    # -------------------------
    # Output
    # -------------------------
    print(f"\nRandom test sample index: {idx}\n")
    print("Predicted probabilities:")
    for label_name, prob in zip(test_dataset.LABELS, probs):
        print(f"  {label_name}: {prob:.4f}")

    print("\nGround truth labels:")
    for label_name, gt in zip(test_dataset.LABELS, label):
        print(f"  {label_name}: {int(gt)}")


if __name__ == "__main__":
    main()

"""
OUTPUT

Random test sample index: 122

Predicted probabilities:
  Atelectasis: 0.6936
  Cardiomegaly: 0.3909
  Consolidation: 0.8258
  Edema: 0.8562
  Enlarged Cardiomediastinum: 0.5988
  Fracture: 0.1028
  Lung Lesion: 0.2068
  Lung Opacity: 0.5846
  Pleural Effusion: 0.9498
  Pleural Other: 0.2405
  Pneumonia: 0.2055
  Pneumothorax: 0.1419
  Support Devices: 0.9114

Ground truth labels:
  Atelectasis: 0
  Cardiomegaly: 0
  Consolidation: 1
  Edema: 0
  Enlarged Cardiomediastinum: 0
  Fracture: 0
  Lung Lesion: 0
  Lung Opacity: 0
  Pleural Effusion: 1
  Pleural Other: 0
  Pneumonia: 0
  Pneumothorax: 0
  Support Devices: 0
"""