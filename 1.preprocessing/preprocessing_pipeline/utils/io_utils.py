"""Shared file I/O helpers."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import pandas as pd


def ensure_directory(path: Path) -> Path:
    """Create a directory if needed and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a JSON document."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def write_pickle(path: Path, payload: Any) -> None:
    """Serialize an object to pickle."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(payload, handle)


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    """Persist a dataframe to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)

