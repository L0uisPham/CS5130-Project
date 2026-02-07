from __future__ import annotations

import torch


def make_criterion(cfg):
    return torch.nn.BCEWithLogitsLoss()
