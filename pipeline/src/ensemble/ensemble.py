from __future__ import annotations

from pathlib import Path

import torch
import yaml
from PIL import Image
from torchvision import transforms
from torch import nn
from torchvision.models import convnext_tiny

def _extract_state_dict(checkpoint: object) -> dict:
    if not isinstance(checkpoint, dict):
        raise RuntimeError("Checkpoint format is unsupported.")

    for key in ("state_dict", "model_state_dict", "model_state", "model", "weights"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return value
    return checkpoint


def _normalize_state_dict_keys(state_dict: dict) -> dict:
    normalized = {}
    for key, value in state_dict.items():
        new_key = key
        for prefix in ("module.", "model."):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
        normalized[new_key] = value
    return normalized


class Ensemble:
    CONV_AUC = {
        "Atelectasis": 0.691884034180245,
        "Cardiomegaly": 0.854334127364679,
        "Consolidation": 0.742082160421844,
        "Edema": 0.841809217921547,
        "Enlarged Cardiomediastinum": 0.680073139286942,
        "Fracture": 0.756709168670716,
        "Lung Lesion": 0.766846277994348,
        "Lung Opacity": 0.731655994269896,
        "No Finding": 0.875933797554707,
        "Pleural Effusion": 0.881332289509101,
        "Pleural Other": 0.811264756917157,
        "Pneumonia": 0.764591546322272,
        "Pneumothorax": 0.860957762266519,
        "Support Devices": 0.874641536403868,
    }

    SWIN_AUC = {
        "Atelectasis": 0.686384024289389,
        "Cardiomegaly": 0.850190976497908,
        "Consolidation": 0.731635152844355,
        "Edema": 0.838320571999253,
        "Enlarged Cardiomediastinum": 0.673289682102493,
        "Fracture": 0.752551173177715,
        "Lung Lesion": 0.775019029452932,
        "Lung Opacity": 0.72936497016274,
        "No Finding": 0.873087758475909,
        "Pleural Effusion": 0.882484651604048,
        "Pleural Other": 0.807022056056859,
        "Pneumonia": 0.759760099263239,
        "Pneumothorax": 0.854932880348299,
        "Support Devices": 0.879525499491342,
    }

    def __init__(
        self,
        config_path: str | Path | None = None,
        weights_dir: str | Path | None = None,
        device: torch.device | None = None,
    ) -> None:
        root = Path(__file__).resolve().parents[3]
        self.config_path = Path(config_path) if config_path else root / "pipeline" / "configs" / "chexpert.yaml"
        self.weights_dir = Path(weights_dir) if weights_dir else root / "weights"
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        with self.config_path.open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

        self.labels = list(config.get("labels", []))
        if not self.labels:
            raise RuntimeError(f"No labels found in config: {self.config_path}")

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        self.conv_model = self._load_model(
            model_name="convnext_t",
            weights_path=self.weights_dir / "convnext.pt",
        )
        self.conv_model_kind = "torchvision"
        self.swin_model = self._load_model(
            model_name="hf_swin_tiny",
            weights_path=self.weights_dir / "swin_best.pt",
        )
        self.swin_model_kind = "huggingface"
        self.selector = self._build_selector()

    def _load_model(self, model_name: str, weights_path: Path):
        if not weights_path.exists():
            raise FileNotFoundError(f"Weights file not found: {weights_path}")

        model = self._build_model(model_name).to(self.device)
        checkpoint = torch.load(weights_path, map_location=self.device)
        state_dict = _normalize_state_dict_keys(_extract_state_dict(checkpoint))
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            raise RuntimeError(
                f"Checkpoint is missing {len(missing)} parameters for {model_name}: "
                + ", ".join(missing[:5])
            )
        if unexpected:
            raise RuntimeError(
                f"Checkpoint has unexpected parameters for {model_name}: "
                + ", ".join(unexpected[:5])
            )
        model.eval()
        return model

    def _build_model(self, model_name: str):
        if model_name == "convnext_t":
            model = convnext_tiny(weights=None)
            in_features = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(in_features, len(self.labels))
            return model

        if model_name == "hf_swin_tiny":
            try:
                from transformers import AutoConfig, AutoModelForImageClassification
            except ImportError as exc:
                raise RuntimeError(
                    "transformers is required to load the Swin ensemble member."
                ) from exc

            config = AutoConfig.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
            config.num_labels = len(self.labels)
            return AutoModelForImageClassification.from_config(config)

        raise RuntimeError(f"Unsupported ensemble model: {model_name}")

    def _build_selector(self) -> dict[str, str]:
        selector = {}
        for pathology in self.labels:
            conv_auc = self.CONV_AUC.get(pathology, float("-inf"))
            swin_auc = self.SWIN_AUC.get(pathology, float("-inf"))
            selector[pathology] = "convnext" if conv_auc >= swin_auc else "swin"
        return selector

    def _predict_model(self, model, model_kind: str, x: torch.Tensor) -> torch.Tensor:
        if model_kind == "huggingface":
            logits = model(pixel_values=x).logits
        elif model_kind == "torchvision":
            logits = model(x)
        else:
            raise RuntimeError(f"Unsupported ensemble model kind: {model_kind}")
        return torch.sigmoid(logits).squeeze(0).cpu()

    def inference_from_path(self, image_path: str | Path) -> tuple[list[str], list[float]]:
        image = Image.open(image_path).convert("RGB")
        x = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            conv_pred = self._predict_model(self.conv_model, self.conv_model_kind, x)
            swin_pred = self._predict_model(self.swin_model, self.swin_model_kind, x)

        final_probs = []
        for index, pathology in enumerate(self.labels):
            selected_model = self.selector[pathology]
            pred = conv_pred[index] if selected_model == "convnext" else swin_pred[index]
            final_probs.append(float(pred.item()))

        return self.labels, final_probs

    def inference(self, image_path: str | Path) -> dict[str, float]:
        labels, probs = self.inference_from_path(image_path)
        return dict(zip(labels, probs))
