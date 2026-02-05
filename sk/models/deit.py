from timm import create_model
from .base import BaseMultiLabelModel


class DeiTModel(BaseMultiLabelModel):
    """
    Wrapper for a DeiT (Data-efficient Image Transformer) model for multi-label classification.

    Attributes:
        model (torch.nn.Module): The underlying DeiT model from TIMM.
    """

    def __init__(self, model_name: str, num_classes: int, pretrained=True):
        """
        Initializes the DeiT model.

        Args:
            model_name (str): Name of the model in TIMM (e.g., "deit_small_patch16_224").
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

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes).
        """

        return self.model(x)

    def freeze_backbone(self):
        """
        Freezes all layers of the model except the classification head.
        Useful for fine-tuning only the head initially.
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
