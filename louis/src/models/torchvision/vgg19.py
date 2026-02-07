from __future__ import annotations

import torch
from torch import nn
from torchvision.models import VGG19_Weights, vgg19

from src.core.types import Batch, ModelOutput
from src.models.base import BaseModelAdapter


class VGG19Adapter(BaseModelAdapter):
    def __init__(self, num_classes: int, pretrained: bool = True) -> None:
        super().__init__()
        weights = VGG19_Weights.DEFAULT if pretrained else None
        self.model = vgg19(weights=weights)
        in_features = self.model.classifier[-1].in_features
        self.model.classifier[-1] = nn.Linear(in_features, num_classes)

    def forward(self, batch: Batch) -> ModelOutput:
        logits = self.model(batch.x)
        return ModelOutput(logits=logits)

    def freeze(self, stage: str) -> None:
        if stage == "head_only":
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.model.classifier.parameters():
                param.requires_grad = True
        elif stage == "last_stage":
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.model.classifier.parameters():
                param.requires_grad = True
            for param in self.model.features[-5:].parameters():
                param.requires_grad = True
        elif stage == "full":
            for param in self.model.parameters():
                param.requires_grad = True
        else:
            raise ValueError(f"Unknown freeze stage: {stage}")
