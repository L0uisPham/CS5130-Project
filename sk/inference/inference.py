import random
from pathlib import Path
import torch
from timm import create_model
from sk.dataset.chexpert import CheXpert


class Inference:
    MODEL_NAME_DICTIONARY = {
        "deit": "deit_small_patch16_224",
        "swin": "swin_tiny_patch4_window7_224",
        "resnet": "resnet34",
        "vgg": "vgg19_bn",
        "efficient": "efficientnet_b4"
    }

    def __init__(self, model_name) -> None:
        self.model_name = model_name.strip().lower()

        if self.model_name not in self.MODEL_NAME_DICTIONARY:
            raise ValueError(
                f"{self.model_name} is not supported. Choose from {list(self.MODEL_NAME_DICTIONARY.keys())}.")

        self.model_path = Path(
            f"sk/tuned_models/best_{self.model_name}_model.pth")
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        self.test_dataset = CheXpert("test")

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.initialize_model()

    def initialize_model(self):
        self.model = create_model(
            self.MODEL_NAME_DICTIONARY[self.model_name],
            pretrained=False,
            num_classes=self.test_dataset.num_classes
        )

        self.model.load_state_dict(torch.load(
            self.model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        print(f"{self.model_name} model is ready for inference!")

    def run_inference(self):
        idx = random.randint(0, len(self.test_dataset) - 1)
        image, label = self.test_dataset[idx]

        image = image.unsqueeze(0).to(self.device)  # type: ignore

        with torch.no_grad():
            logits = self.model(image)
            probs = torch.sigmoid(logits).squeeze(0).cpu()

        print(f"\nRandom test sample index: {idx}\n")
        print("Predicted probabilities:")
        for label_name, prob in zip(self.test_dataset.LABELS, probs):
            print(f"  {label_name}: {prob:.4f}")

        print("\nGround truth labels:")
        for label_name, gt in zip(self.test_dataset.LABELS, label):
            print(f"  {label_name}: {int(gt)}")

        # TODO image not from dataset capability, manual image selection via path


"""
OUTPUT


"""
