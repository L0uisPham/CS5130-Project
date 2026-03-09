from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm

from src.core.types import Batch
from src.models.base import BaseModelAdapter


@torch.no_grad()
def collect_predictions(model: BaseModelAdapter, loader, device, desc: str = "eval") -> Dict:
    model.eval()
    all_y = []
    all_prob = []
    sex_num = []
    age_bin = []

    total_batches = len(loader)
    total_images = len(loader.dataset)
    seen_images = 0

    progress = tqdm(loader, desc=desc, unit="batch", leave=False)
    for batch_idx, batch in enumerate(progress, start=1):
        batch = _to_device(batch, device)
        outputs = model(batch)
        probs = torch.sigmoid(outputs.logits)
        all_y.append(batch.y.detach().cpu())
        all_prob.append(probs.detach().cpu())
        sex_num.append(batch.meta["sex_num"].detach().cpu())
        age_bin.extend(batch.meta["age_bin"])
        seen_images += batch.x.shape[0]
        progress.set_postfix(
            batch=f"{batch_idx}/{total_batches}",
            images=f"{seen_images}/{total_images}",
        )

    y_true = torch.cat(all_y, dim=0).numpy()
    y_prob = torch.cat(all_prob, dim=0).numpy()
    sex_num = torch.cat(sex_num, dim=0).numpy()

    return {
        "y_true": y_true,
        "y_prob": y_prob,
        "sex_num": sex_num,
        "age_bin": age_bin,
    }


def compute_label_auc(y_true, y_prob, label_names: List[str]) -> Dict[str, float]:
    aucs = {}
    for i, name in enumerate(label_names):
        try:
            if len(np.unique(y_true[:, i])) < 2:
                aucs[name] = np.nan
            else:
                aucs[name] = roc_auc_score(y_true[:, i], y_prob[:, i])
        except ValueError:
            aucs[name] = np.nan
    return aucs


def compute_gender_auc(y_true, y_prob, sex_num) -> Dict[str, float]:
    results = {}
    for sex_val in [0, 1]:
        mask = sex_num == sex_val
        if mask.sum() == 0:
            results[f"sex={sex_val}"] = np.nan
            continue
        per_label = []
        for i in range(y_true.shape[1]):
            if len(np.unique(y_true[mask, i])) < 2:
                per_label.append(np.nan)
            else:
                per_label.append(roc_auc_score(y_true[mask, i], y_prob[mask, i]))
        results[f"sex={sex_val}"] = float(np.nanmean(per_label))
    return results


def compute_agebin_auc(y_true, y_prob, age_bin: List[str]) -> Dict[str, float]:
    results = {}
    bins = sorted(set(age_bin), key=_agebin_sort_key)
    age_bin_arr = np.array(age_bin)
    for bin_name in bins:
        mask = age_bin_arr == bin_name
        if mask.sum() == 0:
            results[bin_name] = np.nan
            continue
        per_label = []
        for i in range(y_true.shape[1]):
            if len(np.unique(y_true[mask, i])) < 2:
                per_label.append(np.nan)
            else:
                per_label.append(roc_auc_score(y_true[mask, i], y_prob[mask, i]))
        results[bin_name] = float(np.nanmean(per_label))
    return results


def _agebin_sort_key(label: str) -> int:
    try:
        return int(label.split("-")[0])
    except ValueError:
        return 0


def _to_device(batch: Batch, device) -> Batch:
    batch.x = batch.x.to(device)
    batch.y = batch.y.to(device)
    batch.meta["sex_num"] = batch.meta["sex_num"].to(device)
    batch.meta["age"] = batch.meta["age"].to(device)
    return batch
