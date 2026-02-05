import torch.nn as nn


class BaseMultiLabelModel(nn.Module):
    """
    Abstract base class for multi-label classification models.

    This class defines a common interface for models that support
    freezing and unfreezing of backbone parameters, which is useful
    for staged fine-tuning workflows.
    """

    def freeze_backbone(self):
        """
        Freezes the backbone (feature extractor) of the model.

        This method should be implemented by subclasses and typically
        disables gradient computation for all parameters except the
        classification head.
        """

        raise NotImplementedError

    def unfreeze_backbone(self):
        """
        Unfreezes the backbone of the model.

        This method should be implemented by subclasses and typically
        enables gradient computation for all model parameters.
        """

        raise NotImplementedError

    def get_trainable_params(self):
        """
        Returns an iterator over trainable parameters.

        Only parameters with `requires_grad=True` are included. This is
        useful for constructing optimizers that should operate only on
        unfrozen layers.

        Returns:
            Iterator[nn.Parameter]: Trainable model parameters.
        """

        return filter(lambda p: p.requires_grad, self.parameters())
