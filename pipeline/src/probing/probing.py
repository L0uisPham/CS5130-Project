"""
Linear probing for demographic attributes (sex, age) from frozen
chest-X-ray backbones (ConvNeXt / Swin).

Improvements over the original:
  - Proper feature extraction with a generic hook instead of brittle
    head-surgery that breaks .logits access.
  - Separate validation loop with accuracy (sex) and MAE (age) metrics.
  - Cosine-annealing LR schedule.
  - Early stopping on validation loss.
  - Configurable loss weighting so age MSE doesn't dominate or vanish.
  - Reproducibility seed.
  - Saves best probe weights to disk.
  - Handles missing metadata gracefully (skips samples without sex/age).
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import yaml
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from src.ensemble.ensemble import Ensemble
from src.datasets.loaders import build_dataloaders

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class ProbeConfig:
    epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 1e-4
    age_loss_weight: float = 0.01
    patience: int = 5
    seed: int = 42
    save_dir: str = "probe_weights"
    data_config: str = "pipeline/configs/chexpert.yaml"


# ---------------------------------------------------------------------------
# Feature-extraction wrapper
# ---------------------------------------------------------------------------
class FeatureExtractor(nn.Module):
    """Registers a forward-hook on the penultimate layer so we never need
    to mutate the original model's classifier / head."""

    def __init__(self, backbone: nn.Module, model_name: str):
        super().__init__()
        self.backbone = backbone
        self.model_name = model_name
        self._features: Optional[torch.Tensor] = None
        self._hook = None

        # Freeze everything
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()

        self._register_hook()

    # ------------------------------------------------------------------
    def _register_hook(self):
        """Attach a hook to the layer just before the classification head."""
        target = self._find_penultimate_layer()
        self._hook = target.register_forward_hook(self._hook_fn)
        log.info(
            "Registered feature hook on %s -> %s",
            self.model_name,
            target.__class__.__name__,
        )

    def _hook_fn(self, _module, _input, output):
        self._features = output

    def _find_penultimate_layer(self) -> nn.Module:
        """Heuristic: walk common naming conventions used by timm / HF
        models wrapped in our registry."""
        model = self.backbone

        # If the registry wrapper stores the real model under .model
        inner = getattr(model, "model", model)

        # ConvNeXt (timm): classifier is Sequential(LayerNorm, Flatten, Linear)
        if hasattr(inner, "classifier") and isinstance(inner.classifier, nn.Sequential):
            children = list(inner.classifier.children())
            if len(children) >= 2:
                # Return the second-to-last (e.g. Flatten) so the hook
                # captures the flat feature vector.
                return children[-2]

        # Swin (HF / timm): head is a Linear; norm is just before it
        if hasattr(inner, "norm") and hasattr(inner, "head"):
            return inner.norm

        # Generic fallback: global average pool or adaptive pool
        for name, mod in inner.named_modules():
            if isinstance(mod, (nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d)):
                return mod

        raise RuntimeError(
            f"Cannot locate penultimate layer for {self.model_name}. "
            "Please specify the hook target manually."
        )

    # ------------------------------------------------------------------
    @torch.no_grad()
    def forward(self, batch) -> torch.Tensor:
        """Run the frozen backbone and return the hooked features."""
        self.backbone.eval()
        _ = self.backbone(batch)  # triggers the hook
        feats = self._features
        if feats is None:
            raise RuntimeError("Feature hook did not fire.")
        # Flatten in case of spatial dims (B, C, 1, 1) -> (B, C)
        return feats.flatten(start_dim=1)

    def remove_hook(self):
        if self._hook is not None:
            self._hook.remove()


# ---------------------------------------------------------------------------
# Probe model
# ---------------------------------------------------------------------------
class LinearProbe(nn.Module):
    def __init__(self, feature_extractor: FeatureExtractor, feature_dim: int):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.sex_head = nn.Linear(feature_dim, 1)
        self.age_head = nn.Linear(feature_dim, 1)

    @property
    def name(self) -> str:
        return self.feature_extractor.model_name

    def trainable_parameters(self):
        return list(self.sex_head.parameters()) + list(self.age_head.parameters())

    def forward(self, batch):
        features = self.feature_extractor(batch)
        return self.sex_head(features), self.age_head(features)


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------
def sex_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = (logits.sigmoid() >= 0.5).float()
    return (preds == targets).float().mean().item()


def age_mae(preds: torch.Tensor, targets: torch.Tensor) -> float:
    return (preds - targets).abs().mean().item()


# ---------------------------------------------------------------------------
# Training + validation
# ---------------------------------------------------------------------------
def _extract_targets(batch, device):
    """Pull sex and age tensors from batch metadata.
    Returns (sex, age) each of shape (B, 1) or None when missing."""
    meta = batch.meta
    sex = meta.get("sex_num")
    age = meta.get("age")
    if sex is None or age is None:
        return None, None
    return (
        sex.to(device).float().unsqueeze(1),
        age.to(device).float().unsqueeze(1),
    )


def train_one_epoch(model, loader, optimizer, criterion_sex, criterion_age,
                    device, age_weight):
    model.train()
    # Keep backbone frozen even in .train() mode
    model.feature_extractor.backbone.eval()

    total_loss, total_sex_acc, total_age_mae, n_batches = 0.0, 0.0, 0.0, 0

    for batch in tqdm(loader, desc=f"  train [{model.name}]", leave=False):
        batch.x = batch.x.to(device)
        target_sex, target_age = _extract_targets(batch, device)
        if target_sex is None:
            continue

        sex_logits, age_preds = model(batch)

        loss_sex = criterion_sex(sex_logits, target_sex)
        loss_age = criterion_age(age_preds, target_age)
        loss = loss_sex + age_weight * loss_age

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_sex_acc += sex_accuracy(sex_logits.detach(), target_sex)
        total_age_mae += age_mae(age_preds.detach(), target_age)
        n_batches += 1

    n = max(n_batches, 1)
    return total_loss / n, total_sex_acc / n, total_age_mae / n


@torch.no_grad()
def validate(model, loader, criterion_sex, criterion_age, device, age_weight):
    model.eval()

    total_loss, total_sex_acc, total_age_mae, n_batches = 0.0, 0.0, 0.0, 0

    for batch in tqdm(loader, desc=f"  val   [{model.name}]", leave=False):
        batch.x = batch.x.to(device)
        target_sex, target_age = _extract_targets(batch, device)
        if target_sex is None:
            continue

        sex_logits, age_preds = model(batch)

        loss_sex = criterion_sex(sex_logits, target_sex)
        loss_age = criterion_age(age_preds, target_age)
        loss = loss_sex + age_weight * loss_age

        total_loss += loss.item()
        total_sex_acc += sex_accuracy(sex_logits, target_sex)
        total_age_mae += age_mae(age_preds, target_age)
        n_batches += 1

    n = max(n_batches, 1)
    return total_loss / n, total_sex_acc / n, total_age_mae / n


# ---------------------------------------------------------------------------
# Full training loop with early stopping
# ---------------------------------------------------------------------------
def run_probe(model: LinearProbe, train_loader, val_loader, device,
              cfg: ProbeConfig):
    model.to(device)

    criterion_sex = nn.BCEWithLogitsLoss()
    criterion_age = nn.MSELoss()

    optimizer = optim.AdamW(
        model.trainable_parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    best_path = save_dir / f"{model.name}_probe_best.pt"

    best_val_loss = float("inf")
    patience_counter = 0
    history = []

    for epoch in range(1, cfg.epochs + 1):
        log.info("Epoch %d/%d  [%s]", epoch, cfg.epochs, model.name)

        t_loss, t_sex, t_age = train_one_epoch(
            model, train_loader, optimizer,
            criterion_sex, criterion_age,
            device, cfg.age_loss_weight,
        )
        v_loss, v_sex, v_age = validate(
            model, val_loader,
            criterion_sex, criterion_age,
            device, cfg.age_loss_weight,
        )
        scheduler.step()

        log.info(
            "  train  loss=%.4f  sex_acc=%.3f  age_mae=%.2f", t_loss, t_sex, t_age
        )
        log.info(
            "  val    loss=%.4f  sex_acc=%.3f  age_mae=%.2f", v_loss, v_sex, v_age
        )

        history.append({
            "epoch": epoch,
            "train_loss": t_loss, "train_sex_acc": t_sex, "train_age_mae": t_age,
            "val_loss": v_loss,   "val_sex_acc": v_sex,   "val_age_mae": v_age,
            "lr": optimizer.param_groups[0]["lr"],
        })

        # Checkpoint best
        if v_loss < best_val_loss:
            best_val_loss = v_loss
            patience_counter = 0
            torch.save({
                "sex_head": model.sex_head.state_dict(),
                "age_head": model.age_head.state_dict(),
                "epoch": epoch,
                "val_loss": v_loss,
            }, best_path)
            log.info("  ✓ saved best checkpoint (val_loss=%.4f)", v_loss)
        else:
            patience_counter += 1
            if patience_counter >= cfg.patience:
                log.info("  Early stopping triggered after %d epochs.", epoch)
                break

    # Save training history
    history_path = save_dir / f"{model.name}_probe_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    log.info("Training history saved to %s", history_path)

    return history


# ---------------------------------------------------------------------------
# Detect feature dimension with a dummy forward pass
# ---------------------------------------------------------------------------
@torch.no_grad()
def detect_feature_dim(extractor: FeatureExtractor, sample_batch, device) -> int:
    extractor.to(device)
    sample_batch.x = sample_batch.x[:1].to(device)
    feats = extractor(sample_batch)
    dim = feats.shape[1]
    log.info("Detected feature dim for %s: %d", extractor.model_name, dim)
    return dim


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    cfg = ProbeConfig()

    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    # Data
    with open(cfg.data_config, "r") as f:
        data_cfg = yaml.safe_load(f)
    loaders = build_dataloaders(data_cfg)

    # Models
    ens = Ensemble()

    extractors = {
        "ConvNeXt": FeatureExtractor(ens.conv_model, "ConvNeXt"),
        "Swin":     FeatureExtractor(ens.swin_model, "Swin"),
    }

    # Grab one batch for dimension detection
    sample_batch = next(iter(loaders["train"]))

    for name, extractor in extractors.items():
        feat_dim = detect_feature_dim(extractor, sample_batch, device)
        probe = LinearProbe(extractor, feat_dim)

        log.info("=" * 60)
        log.info("Probing %s  (feature_dim=%d)", name, feat_dim)
        log.info("=" * 60)

        run_probe(probe, loaders["train"], loaders["val"], device, cfg)

        extractor.remove_hook()

    log.info("Done.")


if __name__ == "__main__":
    main()

"""
Output:

22:09:57  INFO      Registered feature hook on ConvNeXt -> Flatten
22:09:57  INFO      Registered feature hook on Swin -> AdaptiveAvgPool1d
22:10:25  INFO      Detected feature dim for ConvNeXt: 768
22:10:25  INFO      ============================================================
22:10:25  INFO      Probing ConvNeXt  (feature_dim=768)
22:10:25  INFO      ============================================================
22:10:25  INFO      Epoch 1/20  [ConvNeXt]
22:35:44  INFO        train  loss=2.7197  sex_acc=0.858  age_mae=11.78                                             
22:35:44  INFO        val    loss=2.3723  sex_acc=0.831  age_mae=11.85
22:35:44  INFO        ✓ saved best checkpoint (val_loss=2.3723)
22:35:44  INFO      Epoch 2/20  [ConvNeXt]
22:50:26  INFO        train  loss=1.7357  sex_acc=0.876  age_mae=9.55                                              
22:50:26  INFO        val    loss=2.5069  sex_acc=0.852  age_mae=12.18
22:50:26  INFO      Epoch 3/20  [ConvNeXt]
23:01:57  INFO        train  loss=1.6713  sex_acc=0.878  age_mae=9.34                                              
23:01:57  INFO        val    loss=2.5729  sex_acc=0.841  age_mae=12.33
23:01:57  INFO      Epoch 4/20  [ConvNeXt]
23:14:31  INFO        train  loss=1.6407  sex_acc=0.878  age_mae=9.24                                              
23:14:31  INFO        val    loss=2.8267  sex_acc=0.844  age_mae=13.09
23:14:31  INFO      Epoch 5/20  [ConvNeXt]
23:33:16  INFO        train  loss=1.6235  sex_acc=0.879  age_mae=9.19                                              
23:33:16  INFO        val    loss=2.8024  sex_acc=0.847  age_mae=13.01
23:33:16  INFO      Epoch 6/20  [ConvNeXt]
23:47:55  INFO        train  loss=1.6153  sex_acc=0.879  age_mae=9.16                                              
23:47:55  INFO        val    loss=2.9532  sex_acc=0.854  age_mae=13.47
23:47:55  INFO        Early stopping triggered after 6 epochs.
23:47:55  INFO      Training history saved to probe_weights\ConvNeXt_probe_history.json
23:47:55  INFO      Detected feature dim for Swin: 768
23:47:55  INFO      ============================================================
23:47:55  INFO      Probing Swin  (feature_dim=768)
23:47:55  INFO      ============================================================
23:47:55  INFO      Epoch 1/20  [Swin]
00:00:21  INFO        train  loss=2.8189  sex_acc=0.852  age_mae=11.94                                             
00:00:21  INFO        val    loss=2.5187  sex_acc=0.870  age_mae=12.11
00:00:21  INFO        ✓ saved best checkpoint (val_loss=2.5187)
00:00:21  INFO      Epoch 2/20  [Swin]
00:18:03  INFO        train  loss=1.7831  sex_acc=0.870  age_mae=9.67                                              
00:18:03  INFO        val    loss=2.4819  sex_acc=0.857  age_mae=11.92
00:18:03  INFO        ✓ saved best checkpoint (val_loss=2.4819)
00:18:03  INFO      Epoch 3/20  [Swin]
00:40:28  INFO        train  loss=1.7252  sex_acc=0.872  age_mae=9.49                                              
00:40:28  INFO        val    loss=2.5038  sex_acc=0.863  age_mae=12.00
00:40:28  INFO      Epoch 4/20  [Swin]
00:52:37  INFO        train  loss=1.7076  sex_acc=0.873  age_mae=9.43                                              
00:52:37  INFO        val    loss=2.6166  sex_acc=0.868  age_mae=12.39
00:52:37  INFO      Epoch 5/20  [Swin]
01:04:31  INFO        train  loss=1.6899  sex_acc=0.874  age_mae=9.37                                                                                                                                                                                                                                                 
01:04:31  INFO        val    loss=2.5402  sex_acc=0.880  age_mae=12.14
01:04:31  INFO      Epoch 6/20  [Swin]
01:16:23  INFO        train  loss=1.6774  sex_acc=0.875  age_mae=9.33                                                                                                                                                                                                                                                 
01:16:23  INFO        val    loss=2.6631  sex_acc=0.850  age_mae=12.43
01:16:23  INFO      Epoch 7/20  [Swin]
01:28:24  INFO        train  loss=1.6736  sex_acc=0.874  age_mae=9.32                                                                                                                                                                                                                                                 
01:28:24  INFO        val    loss=2.5278  sex_acc=0.872  age_mae=12.08
01:28:24  INFO        Early stopping triggered after 7 epochs.
01:28:24  INFO      Training history saved to probe_weights\Swin_probe_history.json
01:28:24  INFO      Done.
"""
