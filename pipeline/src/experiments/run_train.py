from __future__ import annotations

import argparse
import copy
from itertools import product
from pathlib import Path
from typing import Dict

import torch
import yaml

from src.core.io import make_run_dir, save_config
from src.core.seed import set_seed
from src.datasets.loaders import build_dataloaders
from src.models.registry import build_model
from src.training.trainer import fit


def load_yaml(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def merge_dicts(base: Dict, override: Dict) -> Dict:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--prefetch_factor", type=int, default=None)
    parser.add_argument("--from_scratch", action="store_true")
    args = parser.parse_args()

    base_cfg = load_yaml(Path(args.config))

    model_name = args.model
    model_cfg_path = Path("configs") / "models" / f"{model_name}.yaml"
    model_cfg = load_yaml(model_cfg_path)
    merged_cfg = merge_dicts(base_cfg, model_cfg)
    merged_cfg["model"]["name"] = model_name
    merged_cfg["num_classes"] = len(merged_cfg["labels"])
    if args.batch_size is not None:
        merged_cfg.setdefault("data", {})
        merged_cfg["data"]["batch_size"] = args.batch_size
    if args.num_workers is not None:
        merged_cfg.setdefault("data", {})
        merged_cfg["data"]["num_workers"] = args.num_workers
    if args.prefetch_factor is not None:
        merged_cfg.setdefault("data", {})
        merged_cfg["data"]["prefetch_factor"] = args.prefetch_factor

    merged_cfg["seed"] = args.seed

    training_cfg = merged_cfg.get("training", {})
    lr_head_values = training_cfg.get("lr_head", training_cfg.get("lr", 3e-4))
    lr_full_values = training_cfg.get("lr_full", training_cfg.get("lr_finetune", 1e-5))
    weight_decay_values = training_cfg.get("weight_decay", 1e-4)

    def _as_list(value):
        if isinstance(value, (list, tuple)):
            return list(value), True
        return [value], False

    lr_head_values, lr_head_is_list = _as_list(lr_head_values)
    lr_full_values, lr_full_is_list = _as_list(lr_full_values)
    weight_decay_values, weight_decay_is_list = _as_list(weight_decay_values)

    is_grid = lr_head_is_list or lr_full_is_list or weight_decay_is_list
    grid_epochs = int(training_cfg.get("grid_epochs", 10))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _fmt(value) -> str:
        if isinstance(value, float):
            return f"{value:g}"
        return str(value)

    for lr_head, lr_full, weight_decay in product(
        lr_head_values, lr_full_values, weight_decay_values
    ):
        run_cfg = copy.deepcopy(merged_cfg)
        run_cfg.setdefault("training", {})
        run_cfg["training"]["lr_head"] = float(lr_head)
        run_cfg["training"]["lr_full"] = float(lr_full)
        run_cfg["training"]["weight_decay"] = float(weight_decay)
        if is_grid:
            run_cfg["training"]["epochs"] = grid_epochs

        run_name = (
            f"lrh{_fmt(lr_head)}_lrf{_fmt(lr_full)}_wd{_fmt(weight_decay)}_seed{args.seed}"
        )
        run_dir = make_run_dir(run_name=run_name, include_timestamp=True)
        run_cfg["run_id"] = run_dir.name
        save_config(run_dir, run_cfg)

        set_seed(args.seed)
        loaders = build_dataloaders(run_cfg)
        model = build_model(run_cfg).to(device)
        fit(model, loaders, run_cfg, run_dir, device, from_scratch=args.from_scratch)


if __name__ == "__main__":
    main()
