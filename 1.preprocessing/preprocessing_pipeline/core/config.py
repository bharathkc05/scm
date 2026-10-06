"""Configuration loading utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class PipelineConfig:
    """Container for parsed pipeline configuration."""

    path: Path
    values: dict[str, Any]

    @property
    def root_dir(self) -> Path:
        return self.path.parent

    def get(self, *keys: str, default: Any = None) -> Any:
        """Safely fetch a nested configuration value."""
        current: Any = self.values
        for key in keys:
            if not isinstance(current, dict) or key not in current:
                return default
            current = current[key]
        return current

    def resolve_path(self, *keys: str) -> Path:
        """Resolve a configured relative path against the config directory."""
        raw_path = self.get(*keys)
        if raw_path is None:
            raise KeyError(f"Missing configuration path for keys={keys}")
        return (self.root_dir / raw_path).resolve()


def load_config(config_path: Path) -> PipelineConfig:
    """Load YAML configuration from disk."""
    with config_path.open("r", encoding="utf-8") as handle:
        values = yaml.safe_load(handle)
    if not isinstance(values, dict):
        raise ValueError("Configuration file must contain a top-level mapping.")
    return PipelineConfig(path=config_path.resolve(), values=values)

