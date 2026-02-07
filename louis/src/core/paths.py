from __future__ import annotations

from pathlib import Path


def get_project_root() -> Path:
    return Path.cwd().resolve()


def get_data_dir() -> Path:
    return get_project_root() / "data"


def get_processed_dir() -> Path:
    return get_data_dir() / "processed_chexpert"


def get_runs_dir() -> Path:
    return get_project_root() / "runs"

