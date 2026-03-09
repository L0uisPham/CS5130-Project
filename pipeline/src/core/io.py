from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List

import yaml

from src.core.paths import get_runs_dir


def _load_run_config(run_dir: Path) -> Dict:
    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        return {}
    try:
        with config_path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except (OSError, yaml.YAMLError):
        return {}


def _parse_run_id(run_dir: Path) -> str:
    cfg = _load_run_config(run_dir)
    run_id = cfg.get("run_id")
    if isinstance(run_id, str) and run_id:
        return run_id
    return run_dir.name.split("_")[0]


def _parse_model_name(run_dir: Path) -> str:
    cfg = _load_run_config(run_dir)
    model_name = cfg.get("model", {}).get("name") or cfg.get("model_name")
    if isinstance(model_name, str) and model_name:
        return model_name

    run_id = _parse_run_id(run_dir)
    name = run_dir.name
    if "_seed" in name:
        return name[len(run_id) + 1 : name.rfind("_seed")]
    return name[len(run_id) + 1 :]


def _parse_seed(run_dir: Path) -> int:
    cfg = _load_run_config(run_dir)
    seed = cfg.get("seed")
    if isinstance(seed, int):
        return seed
    if isinstance(seed, str) and seed.isdigit():
        return int(seed)

    name = run_dir.name
    if "_seed" not in name:
        return -1
    try:
        return int(name.split("_seed")[-1])
    except ValueError:
        return -1


def make_run_dir(
    model_name: str | None = None,
    seed: int | None = None,
    run_name: str | None = None,
    include_timestamp: bool = False,
) -> Path:
    runs_dir = get_runs_dir()
    runs_dir.mkdir(parents=True, exist_ok=True)
    if run_name is not None:
        base_name = run_name
    else:
        base_name = f"{model_name}_seed{seed}"

    if include_timestamp:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"{run_id}_{base_name}"

    run_dir = runs_dir / base_name
    suffix = 2
    while run_dir.exists():
        run_dir = runs_dir / f"{base_name}_{suffix}"
        suffix += 1
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
