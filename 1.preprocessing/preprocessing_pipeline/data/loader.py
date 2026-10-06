"""Dataset loading and schema validation."""

from __future__ import annotations

from typing import Any

import pandas as pd

from core.config import PipelineConfig
from utils.feature_registry import match_feature_patterns


def load_dataset(config: PipelineConfig) -> pd.DataFrame:
    """Load the input ECG feature dataset."""
    input_path = config.resolve_path("paths", "input_csv")
    return pd.read_csv(input_path)


def validate_loaded_dataset(frame: pd.DataFrame, config: PipelineConfig) -> dict[str, Any]:
    """Validate required columns, row count, and expected schema."""
    required_columns = config.get("dataset", "required_columns", default=[])
    subject_id_column = config.get("dataset", "subject_id_column")
    expected_row_count = config.get("dataset", "expected_row_count")

    missing_required = [column for column in required_columns if column not in frame.columns]
    if missing_required:
        raise ValueError(f"Missing required columns: {missing_required}")
    if subject_id_column not in frame.columns:
        raise ValueError(f"Subject identifier column '{subject_id_column}' is missing.")
    if expected_row_count is not None and len(frame) != expected_row_count:
        raise ValueError(
            f"Unexpected row count. Expected {expected_row_count}, found {len(frame)}."
        )

    integer_columns = match_feature_patterns(
        list(frame.columns),
        config.get("schema", "integer_patterns", default=[]),
    )
    float_columns = match_feature_patterns(
        list(frame.columns),
        config.get("schema", "float_patterns", default=[]),
    )

    dtype_issues: list[str] = []
    for column in integer_columns:
        if not pd.api.types.is_integer_dtype(frame[column].dtype):
            dtype_issues.append(f"{column}: expected integer, found {frame[column].dtype}")
    for column in float_columns:
        if not pd.api.types.is_float_dtype(frame[column].dtype):
            dtype_issues.append(f"{column}: expected float, found {frame[column].dtype}")
    if dtype_issues:
        raise ValueError("Schema validation failed: " + "; ".join(dtype_issues[:10]))

    return {
        "row_count": int(len(frame)),
        "column_count": int(frame.shape[1]),
        "subject_id_column": subject_id_column,
    }

