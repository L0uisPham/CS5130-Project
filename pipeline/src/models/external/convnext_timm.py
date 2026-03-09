from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from src.core.types import Batch, ModelOutput
from src.models.base import BaseModelAdapter


class ConvNeXtTimmAdapter(BaseModelAdapter):
    def __init__(self, model_name: str, num_classes: int, pretrained: bool = True) -> None:
        super().__init__()
        try:
            import timm
        except ImportError as exc:
            raise ImportError("timm is required for ConvNeXtTimmAdapter") from exc
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)

    def forward(self, batch: Batch) -> ModelOutput:
        logits = self.model(batch.x)
        return ModelOutput(logits=logits)

    def freeze(self, stage: str) -> None:
        if stage == "head_only":
            for param in self.model.parameters():
                param.requires_grad = False
            if hasattr(self.model, "head"):
                for param in self.model.head.parameters():
                    param.requires_grad = True
            elif hasattr(self.model, "classifier"):
                for param in self.model.classifier.parameters():
                    param.requires_grad = True
        elif stage == "full":
            for param in self.model.parameters():
                param.requires_grad = True
        else:
            raise ValueError(f"Unknown freeze stage: {stage}")

