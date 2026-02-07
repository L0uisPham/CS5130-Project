from timm import create_model
from sk.model_wrappers.base import BaseMultiLabelModel


class DeiTModel(BaseMultiLabelModel):
    """
    Wrapper for a DeiT (Data-efficient Image Transformer) model for multi-label classification.

    Attributes:
        model (torch.nn.Module): The underlying DeiT model from TIMM.
    """

    def __init__(self, num_classes: int, pretrained=True):
        """
        Initializes the DeiT model.

        Args:
            model_name (str): Name of the model in TIMM (e.g., "deit_small_patch16_224").
            num_classes (int): Number of output classes.
            pretrained (bool): Whether to load pretrained weights.
        """

        super().__init__()
        self.model = create_model(
            "deit_small_patch16_224",
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

    def freeze_backbone_probing(self):
        """Freeze all parameters (including head) for probing."""
        for param in self.model.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """
        Unfreezes all layers of the model for full training.
        """

        for param in self.model.parameters():
            param.requires_grad = True

    def get_features(self, x, pool_type="cls"):
        """
        Extract features from the backbone without the classification head.

        Args:
            x (torch.Tensor): Input images.
            pool_type (str): "cls" for CLS token, "mean" for global average pooling.

        Returns:
            torch.Tensor: Feature embeddings of shape (batch_size, embed_dim)
        """
        # TIMM DeiT returns features with forward_features()
        if hasattr(self.model, "forward_features"):
            # [batch, tokens, dim] or [batch, dim]
            feats = self.model.forward_features(x) # type: ignore
        else:
            feats = self.model(x)

        if pool_type == "cls":
            if feats.dim() == 3:  # [batch, tokens, dim]
                cls_token = feats[:, 0]  # CLS token
                return cls_token
            return feats
        elif pool_type == "mean":
            if feats.dim() == 3:  # [batch, tokens, dim]
                return feats.mean(dim=1)
            return feats
        else:
            raise ValueError(f"Unknown pool_type: {pool_type}")
