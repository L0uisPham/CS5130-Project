# swin.py
from timm import create_model
from sk.model_wrappers.base import BaseMultiLabelModel


class SwinModel(BaseMultiLabelModel):
    """
    Wrapper for a Swin Transformer model for multi-label classification.

    Attributes:
        model (torch.nn.Module): The underlying Swin model from TIMM.
    """

    def __init__(self, model_name: str, num_classes: int, pretrained=True):
        """
        Initializes the Swin model.

        Args:
            model_name (str): Name of the model in TIMM (e.g., "swin_tiny_patch4_window7_224").
            num_classes (int): Number of output classes.
            pretrained (bool): Whether to load pretrained weights.
        """
        super().__init__()
        self.model = create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes
        )

    def forward(self, x):
        """
        Forward pass through the model.
        """
        return self.model(x)

    def freeze_backbone(self):
        """
        Freezes all layers except the classification head.
        For Swin, the head is usually `head`.
        """
        for name, param in self.model.named_parameters():
            if not name.startswith("head"):
                param.requires_grad = False

    def unfreeze_backbone(self):
        """
        Unfreezes all layers of the model for full training.
        """
        for param in self.model.parameters():
            param.requires_grad = True
