"""Configuration loading. Paths are resolved from the repo root, never from the cwd."""
from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "default.yaml"


def load_config(path: str | Path | None = None) -> dict:
    with open(path or DEFAULT_CONFIG, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def resolve(cfg: dict, key: str) -> Path:
    """Absolute path for an entry of ``cfg['paths']``."""
    return ROOT / cfg["paths"][key]
