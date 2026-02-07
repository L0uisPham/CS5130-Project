from sklearn.metrics import roc_auc_score
import torch
from timm import create_model
from sk.dataset.new_chexpert import CheXpert
from sk.model_wrappers.deit import DeiTModel
from sk.model_wrappers.swin import SwinModel
from sk.model_wrappers.resnet import ResNetModel


class Trainer:
    MODEL_NAME_DICTIONARY = {
        "deit": "deit_small_patch16_224",
        "swin": "swin_tiny_patch4_window7_224",
        "resnet": "resnet34",
        "vgg": "vgg_19_imagenet",
        "efficient": "efficientnet-b4"
    }

    NUM_EPOCHS = 10
    FREEZE_EPOCHS = 3
    BATCH_SIZE = 32
    LR_HEAD = 3e-4
    LR_FULL = 1e-5

    def __init__(self, model_name) -> None:
        name_lower = model_name.strip().lower()

        if name_lower not in self.MODEL_NAME_DICTIONARY:
            raise ValueError(
                f"Model '{name_lower}' is not supported. Choose from {list(self.MODEL_NAME_DICTIONARY.keys())}.")

        self.model_name = name_lower

        self.train_dataset = CheXpert("train")
        self.valid_dataset = CheXpert("valid")
        self.test_dataset = CheXpert("test")

        self.train_loader = CheXpert("train").get_loader()
        self.valid_loader = CheXpert("valid").get_loader()
        self.test_loader = CheXpert("test").get_loader()

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.initialize_model()

    def initialize_model(self):
        self.model = create_model(
            self.MODEL_NAME_DICTIONARY[self.model_name],
            pretrained=True,
            num_classes=self.train_dataset.num_classes
        )

    def define_loss_function(self):
        self.label_matrix = self.train_dataset.df[self.train_dataset.LABELS].values

        self.pos_weight = torch.tensor(
            (self.label_matrix.shape[0] - self.label_matrix.sum(axis=0)) /
            (self.label_matrix.sum(axis=0) + 1e-6),
            dtype=torch.float32,
            device=self.device
        )

        self.criterion = torch.nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)

    def forward(self, x):
        return self.model(x)

    def freeze_backbone(self):
        if self.model_name in ["deit", "swin"]:
            starts_with = "head"

        if self.model_name in ["resnet", "vgg", "efficient"]:
            starts_with = "fc"

        for name, param in self.model.named_parameters():
            if not name.startswith(starts_with):
                param.requires_grad = False

    def freeze_backbone_probing(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.model.parameters():
            param.requires_grad = True

    def train_one_epoch(self, loader, criterion, optimizer):
        """
        Trains the model for one epoch.

        Args:
            model (torch.nn.Module): Model to train.
            loader (DataLoader): Training data loader.
            criterion (torch.nn.Module): Loss function.
            optimizer (torch.optim.Optimizer): Optimizer.
            device (torch.device): Device to run training on.

        Returns:
            float: Average loss over the epoch.
        """

        self.model.train()
        total_loss = 0.0

        for images, labels in loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        return total_loss / len(loader)

    @torch.no_grad()
    def validate(self, loader, criterion):
        """
        Evaluates the model on a validation set and computes AUROC for each class.

        Args:
            model (torch.nn.Module): Model to evaluate.
            loader (DataLoader): Validation data loader.
            criterion (torch.nn.Module): Loss function.
            device (torch.device): Device to run evaluation on.
            label_names (list of str): List of class names for AUROC calculation.

        Returns:
            tuple: (average_loss, aurocs) where average_loss is float and
                aurocs is a dict mapping label names to AUROC scores.
        """

        self.model.eval()
        total_loss = 0.0
        all_labels, all_preds = [], []

        for images, labels in loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            all_labels.append(labels.cpu())
            all_preds.append(torch.sigmoid(outputs).cpu())

        all_labels = torch.cat(all_labels).numpy()
        all_preds = torch.cat(all_preds).numpy()

        aurocs = {}
        for i, name in enumerate(self.train_dataset.LABELS):
            try:
                aurocs[name] = roc_auc_score(all_labels[:, i], all_preds[:, i])
            except ValueError:
                aurocs[name] = None

        return total_loss / len(loader), aurocs
