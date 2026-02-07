from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch
from tqdm.auto import tqdm

from src.core.io import append_auc_rows
from src.core.types import Batch
from src.models.base import BaseModelAdapter
from src.training.freeze import apply_freeze, get_stage_for_epoch
from src.training.losses import make_criterion
from src.training.metrics_auc import (
    collect_predictions,
    compute_agebin_auc,
    compute_gender_auc,
    compute_label_auc,
)


def train_one_epoch(
    model: BaseModelAdapter, loader, criterion, optimizer, device, epoch: int, total_epochs: int
) -> float:
    model.train()
    total_loss = 0.0
    total_batches = len(loader)
    total_images = len(loader.dataset)
    seen_images = 0
    progress = tqdm(
        loader,
        desc=f"Epoch {epoch}/{total_epochs} [train]",
        unit="batch",
        leave=False,
    )
    for batch_idx, batch in enumerate(progress, start=1):
        batch = _to_device(batch, device)
        outputs = model(batch)
        loss = criterion(outputs.logits, batch.y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        seen_images += batch.x.shape[0]
        progress.set_postfix(
            loss=f"{loss.item():.4f}",
            batch=f"{batch_idx}/{total_batches}",
            images=f"{seen_images}/{total_images}",
        )
    return total_loss / max(1, len(loader))


def evaluate_auc(
    model: BaseModelAdapter,
    loader,
    device,
    label_names: List[str],
    epoch: int,
    total_epochs: int,
) -> Dict[str, Dict[str, float]]:
    preds = collect_predictions(
        model, loader, device, desc=f"Epoch {epoch}/{total_epochs} [eval]"
    )
    y_true = preds["y_true"]
    y_prob = preds["y_prob"]
    sex_num = preds["sex_num"]
    age_bin = preds["age_bin"]

    return {
        "label": compute_label_auc(y_true, y_prob, label_names),
        "gender": compute_gender_auc(y_true, y_prob, sex_num),
        "age_bin": compute_agebin_auc(y_true, y_prob, age_bin),
    }


def fit(
    model: BaseModelAdapter,
    loaders: Dict[str, object],
    cfg,
    run_dir,
    device,
) -> None:
    label_names = cfg["labels"]
    training_cfg = cfg.get("training", {})
    schedule_cfg = cfg.get("freeze_schedule", {})
    epochs = int(training_cfg.get("epochs", 10))
    weight_decay = float(training_cfg.get("weight_decay", 1e-4))
    lr_head = float(training_cfg.get("lr_head", training_cfg.get("lr", 3e-4)))
    lr_full = float(training_cfg.get("lr_full", training_cfg.get("lr_finetune", 1e-5)))

    criterion = make_criterion(cfg)
    best_val = -float("inf")
    current_stage = None
    optimizer = None

    for epoch in range(epochs):
        stage = get_stage_for_epoch(epoch, schedule_cfg)
        if stage != current_stage:
            apply_freeze(model, stage)
            lr = lr_head if stage == "head_only" else lr_full
            optimizer = torch.optim.AdamW(
                model.parameters_for_optim(), lr=lr, weight_decay=weight_decay
            )
            current_stage = stage

        train_one_epoch(model, loaders["train"], criterion, optimizer, device, epoch + 1, epochs)

        metrics = evaluate_auc(model, loaders["val"], device, label_names, epoch + 1, epochs)
        rows = _metrics_to_rows(metrics, split="val")
        append_auc_rows(run_dir, epoch, rows)

        mean_label_auc = float(np.nanmean(list(metrics["label"].values())))
        if mean_label_auc > best_val:
            best_val = mean_label_auc
            torch.save(model.state_dict(), run_dir / "checkpoints" / "best.pt")

    torch.save(model.state_dict(), run_dir / "checkpoints" / "last.pt")

    test_metrics = evaluate_auc(model, loaders["test"], device, label_names, epochs, epochs)
    test_rows = _metrics_to_rows(test_metrics, split="test")
    append_auc_rows(run_dir, epochs, test_rows)


def _metrics_to_rows(metrics: Dict[str, Dict[str, float]], split: str) -> List[Dict]:
    rows: List[Dict] = []
    for group_type, values in metrics.items():
        for group_name, auc in values.items():
            rows.append(
                {
                    "split": split,
                    "group_type": group_type,
                    "group_name": group_name,
                    "auc": auc,
                }
            )
    return rows


def _to_device(batch: Batch, device) -> Batch:
    batch.x = batch.x.to(device)
    batch.y = batch.y.to(device)
    batch.meta["sex_num"] = batch.meta["sex_num"].to(device)
    batch.meta["age"] = batch.meta["age"].to(device)
    return batch
