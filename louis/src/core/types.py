from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch


@dataclass
class Batch:
    x: torch.Tensor
    y: torch.Tensor
    meta: Dict[str, object]


@dataclass
class ModelOutput:
    logits: torch.Tensor
