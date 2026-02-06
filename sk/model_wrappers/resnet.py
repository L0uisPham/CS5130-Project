from timm import create_model
from sk.model_wrappers.base import BaseMultiLabelModel

class ResNetModel(BaseMultiLabelModel):
    """
    Wrapper for a ResNet model for multi-label classification.

    Attributes:
        model (torch.nn.Module): The underlying ResNet model from TIMM.
    """

    def __init__(self, model_name: str, num_classes: int, pretrained=True):
        """
        Initializes the ResNet model.

        Args:
            model_name (str): Name of the ResNet model in TIMM (e.g., "resnet34", "resnet50").
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
        return self.model(x)

    def freeze_backbone(self):
        """
        Freezes all layers except the classification head (fc layer).
        """
        for name, param in self.model.named_parameters():
            if not name.startswith("fc"):
                param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.model.parameters():
            param.requires_grad = True
