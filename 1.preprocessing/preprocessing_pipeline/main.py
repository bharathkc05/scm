"""Entry point for the ECG preprocessing pipeline."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from cleaning.deduplication import remove_exact_duplicates
from cleaning.feature_removal import get_features_to_remove, remove_features
from cleaning.outlier_treatment import apply_winsorization, fit_winsorization_bounds
from cleaning.scaling import apply_scalers, fit_scalers
from cleaning.transformation import apply_transformations, fit_transformations
from cleaning.type_validation import run_data_validation
from core.config import load_config
from core.logger import setup_logger
from data.exporter import export_datasets, export_metadata, export_parameters
from data.loader import load_dataset, validate_loaded_dataset
from data.splitter import split_by_subject
from utils.feature_registry import build_feature_groups
from validation.integrity_checks import validate_processed_splits
from validation.redundancy_checks import run_redundancy_checks


def main() -> None:
    """Run the full preprocessing pipeline."""
    config = load_config(Path(__file__).resolve().parent / "config.yaml")
    output_dir = config.resolve_path("paths", "output_dir")
    logger, log_path = setup_logger(output_dir / "logs")

    logger.info("START PIPELINE")

    frame = load_dataset(config)
    load_summary = validate_loaded_dataset(frame, config)
    logger.info("Dataset loaded | shape=%s", frame.shape)

    deduplicated_frame, duplicate_count = remove_exact_duplicates(frame)
    logger.info("Removed duplicate rows | removed=%s", duplicate_count)

    original_columns = deduplicated_frame.columns.tolist()
    features_to_remove = get_features_to_remove(
        original_columns,
        config.get("feature_removal", "explicit", default=[]),
        config.get("feature_removal", "regex_patterns", default=[]),
    )
    reduced_frame = remove_features(deduplicated_frame, features_to_remove)
    logger.info("Removed features | removed=%s", len(features_to_remove))

    pre_validation = run_data_validation(reduced_frame, config)
    if pre_validation["impossible_values"]["total"] > 0:
        logger.warning(
            "Impossible values found | total=%s | affected_features=%s",
            pre_validation["impossible_values"]["total"],
            list(pre_validation["impossible_values"]["by_feature"].keys())[:10],
        )
    logger.info("Validation passed | stage=pre_split")

    feature_groups = build_feature_groups(
        list(reduced_frame.columns),
        config.get("feature_groups"),
    )

    splits, split_row_summary = split_by_subject(reduced_frame, config)
    logger.info(
        "Split completed | train=%s validation=%s test=%s",
        splits["train"].shape,
        splits["validation"].shape,
        splits["test"].shape,
    )

    winsorization_bounds = fit_winsorization_bounds(splits["train"], feature_groups, config)
    winsorized_splits = {
        split_name: apply_winsorization(split_frame, winsorization_bounds)
        for split_name, split_frame in splits.items()
    }
    logger.info("Winsorization fitted and applied")

    transformation_artifact = fit_transformations(
        winsorized_splits["train"],
        feature_groups,
        config,
    )
    transformed_splits = {
        split_name: apply_transformations(split_frame, feature_groups, transformation_artifact)
        for split_name, split_frame in winsorized_splits.items()
    }
    logger.info("Transformation completed")

    scaler_artifact = fit_scalers(transformed_splits["train"], feature_groups, config)
    scaled_splits = {
        split_name: apply_scalers(split_frame, scaler_artifact)
        for split_name, split_frame in transformed_splits.items()
    }
    logger.info("Scaling completed")

    expected_feature_count = reduced_frame.shape[1] - 1
    post_validation = validate_processed_splits(
        scaled_splits,
        config,
        expected_feature_count=expected_feature_count,
    )
    redundancy_summary = run_redundancy_checks(scaled_splits["train"], config)
    logger.info("Validation passed | stage=post_processing")

    split_summary = _build_split_summary(scaled_splits, config)
    retained_features = [column for column in reduced_frame.columns if column != "SubjectID"]
    summary_payload = {
        "input": load_summary,
        "duplicate_rows_removed": duplicate_count,
        "removed_feature_count": len(features_to_remove),
        "removed_features": features_to_remove,
        "retained_feature_count": len(retained_features),
        "split_row_summary": split_row_summary,
        "pre_validation": pre_validation,
        "post_validation": post_validation,
        "redundancy_checks": redundancy_summary,
        "log_file": str(log_path),
    }

    export_datasets(output_dir, scaled_splits)
    export_metadata(
        output_dir,
        removed_features=features_to_remove,
        retained_features=retained_features,
        split_summary=split_summary,
        summary_payload=summary_payload,
    )
    export_parameters(
        output_dir,
        winsorization_bounds=winsorization_bounds,
        transformation_artifact=transformation_artifact,
        scaler_artifact=scaler_artifact,
    )
    logger.info("Export completed")
    logger.info("Pipeline finished successfully")


def _build_split_summary(
    splits: dict[str, pd.DataFrame],
    config,
) -> pd.DataFrame:
    subject_id_column = config.get("dataset", "subject_id_column")
    rows = []
    for split_name, frame in splits.items():
        rows.append(
            {
                "split": split_name,
                "row_count": int(frame.shape[0]),
                "subject_count": int(frame[subject_id_column].nunique()),
                "feature_count": int(frame.shape[1] - 1),
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()
