import random
import torch
from torchvision import transforms
from sk.datasets.chexpert import CheXpertDataset
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
