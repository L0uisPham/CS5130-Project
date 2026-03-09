from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable

import torch
from torch import nn

from src.core.types import Batch, ModelOutput


class BaseModelAdapter(nn.Module, ABC):
    @abstractmethod
    def forward(self, batch: Batch) -> ModelOutput:
        raise NotImplementedError

    @abstractmethod
    def freeze(self, stage: str) -> None:
        raise NotImplementedError

    def parameters_for_optim(self) -> Iterable[torch.nn.Parameter]:
        return filter(lambda p: p.requires_grad, self.parameters())
