from __future__ import annotations

from src.models.base import BaseModelAdapter


def get_stage_for_epoch(epoch: int, schedule_cfg) -> str:
    warmup = int(schedule_cfg.get("warmup_epochs", 0))
    if epoch < warmup:
        return "head_only"
    return "full"


def apply_freeze(model: BaseModelAdapter, stage: str) -> None:
    model.freeze(stage)
