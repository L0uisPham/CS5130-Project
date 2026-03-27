import pandas as pd
import torch
import yaml
from torch import nn
from torchvision import transforms
from pathlib import Path
from PIL import Image

from src.core.types import Batch
from src.models.registry import build_model


class Ensemble:

    def __init__(self, config_path: str = "pipeline/configs/chexpert.yaml") -> None:

        self.config_path = Path(config_path)

        with open(self.config_path, "r", encoding="utf-8") as f:
            self.labels = yaml.safe_load(f)["labels"]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.conv_model = self.load_model("convnext")
        self.swin_model = self.load_model("swin")

        # Example ROC AUCs (replace with yours)
        self.conv_auc = {
            'Atelectasis':0.691884034180245,
            'Cardiomegaly':0.854334127364679,
            'Consolidation':0.742082160421844,
            'Edema':0.841809217921547,
            'Enlarged Cardiomediastinum':0.680073139286942,
            'Fracture':0.756709168670716,
            'Lung Lesion':0.766846277994348,
            'Lung Opacity':0.731655994269896,
            'No Finding':0.875933797554707,
            'Pleural Effusion':0.881332289509101,
            'Pleural Other':0.811264756917157,
            'Pneumonia':0.764591546322272,
            'Pneumothorax':0.860957762266519,
            'Support Devices':0.874641536403868
        }

        self.swin_auc = {
            'Atelectasis':0.686384024289389,
            'Cardiomegaly':0.850190976497908,
            'Consolidation':0.731635152844355,
            'Edema':0.838320571999253,
            'Enlarged Cardiomediastinum':0.673289682102493,
            'Fracture':0.752551173177715,
            'Lung Lesion':0.775019029452932,
            'Lung Opacity':0.72936497016274,
            'No Finding':0.873087758475909,
            'Pleural Effusion':0.882484651604048,
            'Pleural Other':0.807022056056859,
            'Pneumonia':0.759760099263239,
            'Pneumothorax':0.854932880348299,
            'Support Devices':0.879525499491342
        }

        self.selector = self.build_selector()

        self.transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485,0.456,0.406],
                std=[0.229,0.224,0.225]
            )
        ])


    def load_model(self, model_name: str):

        if model_name == "convnext":

            cfg = {
                "model":{"name":"convnext_t"},
                "labels": self.labels
            }

            weights_path = Path("weights/convnext.pt")

        elif model_name == "swin":

            cfg = {
                "model":{"name":"hf_swin_tiny"},
                "labels": self.labels
            }

            weights_path = Path("weights/swin.pt")


        model = build_model(cfg)

        checkpoint = torch.load(weights_path,
                                map_location=self.device)

        state_dict = checkpoint.get("model_state", checkpoint)

        model.load_state_dict(state_dict)

        return model.to(self.device).eval()


    def build_selector(self):

        selector = {}

        for pathology in self.labels:

            if self.conv_auc[pathology] >= self.swin_auc[pathology]:

                selector[pathology] = {
                    "convnext":1,
                    "swin":0
                }

            else:

                selector[pathology] = {
                    "convnext":0,
                    "swin":1
                }

        return selector


    def forward_model(self, model, x):

        try:
            out = model(x)

        except:
            out = model({"x":x})

        if hasattr(out,"logits"):
            out = out.logits

        return torch.sigmoid(out)


    def inference(self, image_path):

        image = Image.open(image_path).convert("RGB")

        x = self.transform(image).unsqueeze(0).to(self.device)

        batch = Batch(
            x=x,
            y=torch.empty(0),           # placeholder tensor for labels
            meta={}                      # empty dictionary
        )

        with torch.no_grad():

            conv_out = self.conv_model(batch)
            swin_out = self.swin_model(batch)

            conv_pred = torch.sigmoid(
                conv_out.logits
            ).cpu().squeeze()

            swin_pred = torch.sigmoid(
                swin_out.logits
            ).cpu().squeeze()


        final_preds = {}

        for i, pathology in enumerate(self.labels):

            if self.selector[pathology]["convnext"]:

                final_preds[pathology] = conv_pred[i].item()

            else:

                final_preds[pathology] = swin_pred[i].item()


        return final_preds
    
e = Ensemble()

results = e.inference("test_xray.jpg")

print(results)