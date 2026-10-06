"""Data validation for missingness, infinities, dtypes, and value rules."""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

from core.config import PipelineConfig
from utils.feature_registry import match_feature_patterns


def run_data_validation(
    frame: pd.DataFrame,
    config: PipelineConfig,
) -> dict[str, Any]:
    """Run configured validation checks and return a compact summary."""
    numeric_frame = frame.select_dtypes(include=["number"])
    nan_count = int(frame.isna().sum().sum())
    inf_count = int(np.isinf(numeric_frame.to_numpy()).sum())

    invalid_dtype_columns = _find_invalid_dtype_columns(frame, config)
    impossible_values = _find_impossible_values(frame, config)
    sentinel_hits = _find_sentinel_values(frame, config)

    if nan_count > 0:
        raise ValueError(f"NaN detected: {nan_count} values.")
    if inf_count > 0:
        raise ValueError(f"Inf detected: {inf_count} values.")
    if invalid_dtype_columns:
        raise ValueError(f"Unexpected datatype columns detected: {invalid_dtype_columns[:10]}")
    if sentinel_hits["total"] > 0:
        raise ValueError(f"Sentinel values found: {sentinel_hits['total']}")

    return {
        "nan_count": nan_count,
        "inf_count": inf_count,
        "invalid_dtype_columns": invalid_dtype_columns,
        "impossible_values": impossible_values,
        "sentinel_hits": sentinel_hits,
    }


def _find_invalid_dtype_columns(frame: pd.DataFrame, config: PipelineConfig) -> list[str]:
    columns = list(frame.columns)
    integer_columns = match_feature_patterns(
        columns, config.get("schema", "integer_patterns", default=[])
    )
    float_columns = match_feature_patterns(
        columns, config.get("schema", "float_patterns", default=[])
    )
    issues: list[str] = []
    for column in integer_columns:
        if not pd.api.types.is_integer_dtype(frame[column].dtype):
            issues.append(column)
    for column in float_columns:
        if not pd.api.types.is_float_dtype(frame[column].dtype):
            issues.append(column)
    return issues


def _find_impossible_values(frame: pd.DataFrame, config: PipelineConfig) -> dict[str, Any]:
    non_negative_patterns = config.get("validation", "non_negative_patterns", default=[])
    physiological_limits = config.get("validation", "physiological_limits", default={})

    findings: dict[str, int] = {}
    for pattern in non_negative_patterns:
        matching_columns = [column for column in frame.columns if re.match(pattern, column)]
        for column in matching_columns:
            count = int((frame[column] < 0).sum())
            if count > 0:
                findings[column] = findings.get(column, 0) + count
    for column, limits in physiological_limits.items():
        if column not in frame.columns:
            continue
        lower = limits["min"]
        upper = limits["max"]
        count = int(((frame[column] < lower) | (frame[column] > upper)).sum())
        if count > 0:
            findings[column] = findings.get(column, 0) + count
    return {"total": int(sum(findings.values())), "by_feature": findings}


def _find_sentinel_values(frame: pd.DataFrame, config: PipelineConfig) -> dict[str, Any]:
    sentinel_values = set(config.get("validation", "sentinel_values", default=[]))
    findings: dict[str, int] = {}
    for column in frame.select_dtypes(include=["number"]).columns:
        count = int(frame[column].isin(sentinel_values).sum())
        if count > 0:
            findings[column] = count
    return {"total": int(sum(findings.values())), "by_feature": findings}
