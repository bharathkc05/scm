"""Post-processing integrity checks."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from core.config import PipelineConfig


def validate_processed_splits(
    splits: dict[str, pd.DataFrame],
    config: PipelineConfig,
    expected_feature_count: int,
) -> dict[str, Any]:
    """Validate final splits for core integrity requirements."""
    subject_id_column = config.get("dataset", "subject_id_column")

    summary: dict[str, Any] = {
        "duplicate_rows": {},
        "nan_count": {},
        "inf_count": {},
        "feature_count": {},
    }

    for split_name, frame in splits.items():
        summary["duplicate_rows"][split_name] = int(frame.duplicated().sum())
        summary["nan_count"][split_name] = int(frame.isna().sum().sum())
        summary["inf_count"][split_name] = int(
            np.isinf(frame.select_dtypes(include=["number"]).to_numpy()).sum()
        )
        summary["feature_count"][split_name] = int(frame.shape[1] - 1)

        if summary["duplicate_rows"][split_name] > 0:
            raise ValueError(f"Duplicate rows found in {split_name} split.")
        if summary["nan_count"][split_name] > 0:
            raise ValueError(f"NaN detected in {split_name} split.")
        if summary["inf_count"][split_name] > 0:
            raise ValueError(f"Inf detected in {split_name} split.")
        if summary["feature_count"][split_name] != expected_feature_count:
            raise ValueError(
                f"Unexpected feature count in {split_name}. "
                f"Expected {expected_feature_count}, found {summary['feature_count'][split_name]}."
            )

    train_ids = set(splits["train"][subject_id_column].tolist())
    validation_ids = set(splits["validation"][subject_id_column].tolist())
    test_ids = set(splits["test"][subject_id_column].tolist())
    if train_ids & validation_ids or train_ids & test_ids or validation_ids & test_ids:
        raise ValueError("Duplicate SubjectIDs detected across splits.")

    summary["split_subject_overlap"] = 0
    return summary

