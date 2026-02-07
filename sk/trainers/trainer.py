import os
from sklearn.metrics import roc_auc_score
import torch
from timm import create_model
from sk.dataset.new_chexpert import CheXpert


class Trainer:
    MODEL_NAME_DICTIONARY = {
        "deit": "deit_small_patch16_224",
        "swin": "swin_tiny_patch4_window7_224",
        "resnet": "resnet34",
        "vgg": "vgg19_bn",
        "efficient": "efficientnet_b4"
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

        self.train_loader = self.train_dataset.get_loader()
        self.valid_loader = self.valid_dataset.get_loader()
        self.test_loader = self.test_dataset.get_loader()

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.initialize_model()
        self.loss_function_and_optimizer()

        print(f"{self.model_name} model is ready for .training_loop()")

    def initialize_model(self):
        self.model = create_model(
            self.MODEL_NAME_DICTIONARY[self.model_name],
            pretrained=True,
            num_classes=self.train_dataset.num_classes
        ).to(self.device)

    def loss_function_and_optimizer(self):
        self.label_matrix = torch.tensor(
            self.train_dataset.df[self.train_dataset.LABELS].values,
            dtype=torch.float32,
            device=self.device
        )

        self.pos_weight = ((self.label_matrix.shape[0] - self.label_matrix.sum(
            dim=0)) / (self.label_matrix.sum(dim=0) + 1e-6)).to(self.device)

        self.criterion = torch.nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)

        self.freeze_backbone()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.LR_FULL)
        self.best_val_loss = float("inf")

    def forward(self, x):
        return self.model(x)

    def freeze_backbone(self):
        if self.model_name in ["deit", "swin"]:
            head_name = "head"
        elif self.model_name == "resnet":
            head_name = "fc"
        elif self.model_name in ["vgg", "efficient"]:
            head_name = "classifier"
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")

        for name, param in self.model.named_parameters():
            param.requires_grad = name.startswith(head_name)

    def unfreeze_backbone(self):
        for param in self.model.parameters():
            param.requires_grad = True

    def train_one_epoch(self, loader):
        self.model.train()
        total_loss = 0.0

        for images, labels in loader:
            images = images.to(self.device)
            labels = labels.float().to(self.device)

            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(loader)

    @torch.no_grad()
    def validate(self, loader):
        self.model.eval()
        total_loss = 0.0
        all_labels, all_preds = [], []

        for images, labels in loader:
            images = images.to(self.device)
            labels = labels.float().to(self.device)

            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

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

    def training_loop(self):
        os.makedirs("sk/tuned_models", exist_ok=True)
        orig_path = f"sk/tuned_models/best_{self.model_name}_model.pth"

        for epoch in range(self.NUM_EPOCHS):
            print(f"\nEpoch {epoch + 1}/{self.NUM_EPOCHS}")

            if epoch == 0:
                self.freeze_backbone()
                self.optimizer = torch.optim.AdamW(
                    filter(lambda p: p.requires_grad, self.model.parameters()),
                    lr=self.LR_HEAD
                )

            if epoch == self.FREEZE_EPOCHS:
                self.unfreeze_backbone()
                self.optimizer = torch.optim.AdamW(
                    self.model.parameters(),
                    lr=self.LR_FULL
                )

            train_loss = self.train_one_epoch(self.train_loader)

            val_loss, aurocs = self.validate(self.valid_loader)

            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            for k, v in aurocs.items():
                if v is not None:
                    print(f"  {k}: {v:.4f}")

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save(self.model.state_dict(), orig_path)
                print("Model saved!")

        print("\nTraining complete")

        print("\nEvaluating on test_strat set...")

        self.model.load_state_dict(torch.load(
            orig_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        test_loss, test_aurocs = self.validate(self.test_loader)

        print(f"\nTest Loss: {test_loss:.4f}")
        print("Test AUROC scores per pathology:")
        for k, v in test_aurocs.items():
            if v is not None:
                print(f"  {k}: {v:.4f}")

        safe_loss = int(test_loss * 10000)
        new_path = f"sk/tuned_models/best_{self.model_name}_model_test_{safe_loss}.pth"
        os.rename(orig_path, new_path)

        print(f"Model file renamed to {new_path}!")
