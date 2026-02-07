from __future__ import annotations

import argparse
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

    run_dir = make_run_dir(model_name=model_name, seed=args.seed)
    save_config(run_dir, merged_cfg)

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders = build_dataloaders(merged_cfg)
    model = build_model(merged_cfg).to(device)

    fit(model, loaders, merged_cfg, run_dir, device)


if __name__ == "__main__":
    main()
