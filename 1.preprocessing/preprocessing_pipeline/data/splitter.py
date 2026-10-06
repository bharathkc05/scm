"""Subject-level train/validation/test splitting."""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from core.config import PipelineConfig


def split_by_subject(
    frame: pd.DataFrame,
    config: PipelineConfig,
) -> tuple[dict[str, pd.DataFrame], dict[str, int]]:
    """Split a dataframe by subject identifier without row-level leakage."""
    subject_id_column = config.get("dataset", "subject_id_column")
    train_ratio = float(config.get("split", "train_ratio"))
    validation_ratio = float(config.get("split", "validation_ratio"))
    test_ratio = float(config.get("split", "test_ratio"))
    random_seed = int(config.get("split", "random_seed"))

    total_ratio = train_ratio + validation_ratio + test_ratio
    if round(total_ratio, 10) != 1.0:
        raise ValueError("Train, validation, and test ratios must sum to 1.0.")

    unique_subjects = frame[subject_id_column].drop_duplicates().to_numpy()
    train_subjects, holdout_subjects = train_test_split(
        unique_subjects,
        test_size=(1.0 - train_ratio),
        random_state=random_seed,
        shuffle=True,
    )

    validation_share = validation_ratio / (validation_ratio + test_ratio)
    validation_subjects, test_subjects = train_test_split(
        holdout_subjects,
        train_size=validation_share,
        random_state=random_seed,
        shuffle=True,
    )

    split_subjects = {
        "train": set(train_subjects.tolist()),
        "validation": set(validation_subjects.tolist()),
        "test": set(test_subjects.tolist()),
    }

    splits = {
        split_name: frame[frame[subject_id_column].isin(subjects)].copy()
        for split_name, subjects in split_subjects.items()
    }
    summary = {
        split_name: int(split_frame.shape[0])
        for split_name, split_frame in splits.items()
    }
    return splits, summary

