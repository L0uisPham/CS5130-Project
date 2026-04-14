import random
from pathlib import Path
from typing import List, Tuple

import torch
from timm import create_model
from PIL import Image
from torchvision import transforms

from scripts.sk.dataset.chexpert import CheXpert


SK_ROOT = Path(__file__).resolve().parents[1]

# Same eval transform as CheXpert test set (Resize 224, ToTensor, ImageNet normalize)
EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


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

        self.model_path = SK_ROOT / "tuned_models" / f"best_{self.model_name}_model.pth"
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

    def inference_from_path(self, image_path: str) -> Tuple[List[str], List[float]]:
        """
        Run model on a single image file. Returns (label_names, probabilities) for use
        with LLMollama or other pipelines. Uses first 14 labels to match CheXpert 14-condition outputs.
        """
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        image = Image.open(path).convert("RGB")
        image = EVAL_TRANSFORM(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(image)
            probs = torch.sigmoid(logits).squeeze(0).cpu()

        labels = list(self.test_dataset.LABELS)
        # Use first 14 for CheXpert 14-condition compatibility (e.g. LLMollama)
        n = min(14, len(labels), probs.numel())
        return labels[:n], probs[:n].tolist()


"""
OUTPUT


"""
