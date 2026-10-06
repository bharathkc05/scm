"""Artifact export helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from core.constants import DATASET_FILE_NAMES, METADATA_FILE_NAMES, PARAMETER_FILE_NAMES
from utils.io_utils import write_csv, write_json, write_pickle


def export_datasets(base_output_dir: Path, splits: dict[str, pd.DataFrame]) -> None:
    """Export preprocessed splits as CSV files."""
    datasets_dir = base_output_dir / "datasets"
    for split_name, frame in splits.items():
        write_csv(datasets_dir / DATASET_FILE_NAMES[split_name], frame)


def export_metadata(
    base_output_dir: Path,
    removed_features: list[str],
    retained_features: list[str],
    split_summary: pd.DataFrame,
    summary_payload: dict[str, Any],
) -> None:
    """Export metadata files."""
    metadata_dir = base_output_dir / "metadata"
    write_csv(
        metadata_dir / METADATA_FILE_NAMES["removed_features"],
        pd.DataFrame({"feature": removed_features}),
    )
    write_csv(
        metadata_dir / METADATA_FILE_NAMES["retained_features"],
        pd.DataFrame({"feature": retained_features}),
    )
    write_csv(metadata_dir / METADATA_FILE_NAMES["split_summary"], split_summary)
    write_json(metadata_dir / METADATA_FILE_NAMES["summary"], summary_payload)


def export_parameters(
    base_output_dir: Path,
    winsorization_bounds: dict[str, dict[str, float]],
    transformation_artifact: Any,
    scaler_artifact: Any,
) -> None:
    """Export fitted preprocessing parameters."""
    parameters_dir = base_output_dir / "parameters"
    write_json(parameters_dir / PARAMETER_FILE_NAMES["winsorization"], winsorization_bounds)
    write_pickle(parameters_dir / PARAMETER_FILE_NAMES["transformers"], transformation_artifact)
    write_pickle(parameters_dir / PARAMETER_FILE_NAMES["scaler"], scaler_artifact)

