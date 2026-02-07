from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List

import yaml

from src.core.paths import get_runs_dir


def _parse_run_id(run_dir: Path) -> str:
    return run_dir.name.split("_")[0]


def _parse_model_name(run_dir: Path) -> str:
    run_id = _parse_run_id(run_dir)
    name = run_dir.name
    if "_seed" in name:
        return name[len(run_id) + 1 : name.rfind("_seed")]
    return name[len(run_id) + 1 :]


def _parse_seed(run_dir: Path) -> int:
    name = run_dir.name
    if "_seed" not in name:
        return -1
    try:
        return int(name.split("_seed")[-1])
    except ValueError:
        return -1


def make_run_dir(model_name: str, seed: int) -> Path:
    runs_dir = get_runs_dir()
    runs_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = runs_dir / f"{run_id}_{model_name}_seed{seed}"
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    return run_dir


def save_config(run_dir: Path, merged_cfg: Dict) -> None:
    config_path = run_dir / "config.yaml"
    with config_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(merged_cfg, f, sort_keys=False)


def append_auc_rows(run_dir: Path, epoch: int, rows: List[Dict]) -> None:
    csv_path = run_dir / "auc_by_epoch.csv"
    fieldnames = [
        "epoch",
        "split",
        "group_type",
        "group_name",
        "auc",
        "model_name",
        "seed",
        "run_id",
    ]
    run_id = _parse_run_id(run_dir)
    model_name = _parse_model_name(run_dir)
    seed = _parse_seed(run_dir)

    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for row in rows:
            payload = dict(row)
            payload.setdefault("epoch", epoch)
            payload.setdefault("model_name", model_name)
            payload.setdefault("seed", seed)
            payload.setdefault("run_id", run_id)
            writer.writerow(payload)
