from __future__ import annotations

from src.core.types import Batch, ModelOutput
from src.models.base import BaseModelAdapter


class HfImageClassifierAdapter(BaseModelAdapter):
    def __init__(self, model_name: str, num_classes: int, pretrained: bool = True) -> None:
        super().__init__()
        try:
            from transformers import AutoModelForImageClassification
        except ImportError as exc:
            raise ImportError("transformers is required for HuggingFace adapters") from exc

        self.model = AutoModelForImageClassification.from_pretrained(
            model_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        )

    def forward(self, batch: Batch) -> ModelOutput:
        outputs = self.model(batch.x)
        return ModelOutput(logits=outputs.logits)

    def freeze(self, stage: str) -> None:
        if stage == "head_only":
            for param in self.model.parameters():
                param.requires_grad = False
            if hasattr(self.model, "classifier"):
                for param in self.model.classifier.parameters():
                    param.requires_grad = True
            elif hasattr(self.model, "head"):
                for param in self.model.head.parameters():
                    param.requires_grad = True
        elif stage == "full":
            for param in self.model.parameters():
                param.requires_grad = True
        else:
            raise ValueError(f"Unknown freeze stage: {stage}")

