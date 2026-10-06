"""Scaling fit/apply logic."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler

from core.config import PipelineConfig


def fit_scalers(
    train_frame: pd.DataFrame,
    feature_groups: dict[str, list[str]],
    config: PipelineConfig,
) -> dict[str, Any]:
    """Fit configured scalers on the training split only."""
    by_group = config.get("scaling", "by_group", default={})
    scalers: dict[str, Any] = {}

    for group_name, scaler_name in by_group.items():
        columns = [column for column in feature_groups.get(group_name, []) if column in train_frame]
        if not columns:
            continue
        scaler = _build_scaler(scaler_name)
        scaler.fit(train_frame[columns])
        scalers[group_name] = {"columns": columns, "scaler_name": scaler_name, "scaler": scaler}
    return scalers


def apply_scalers(frame: pd.DataFrame, scaler_artifact: dict[str, Any]) -> pd.DataFrame:
    """Apply fitted scalers to a dataframe."""
    transformed = frame.copy()
    for group_payload in scaler_artifact.values():
        columns = group_payload["columns"]
        scaler = group_payload["scaler"]
        transformed[columns] = scaler.transform(transformed[columns])
    return transformed


def _build_scaler(name: str) -> StandardScaler | RobustScaler:
    if name == "standard":
        return StandardScaler()
    if name == "robust":
        return RobustScaler()
    raise ValueError(f"Unsupported scaler type: {name}")

