"""Feature transformation fit/apply logic."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer

from core.config import PipelineConfig


def fit_transformations(
    train_frame: pd.DataFrame,
    feature_groups: dict[str, list[str]],
    config: PipelineConfig,
) -> dict[str, Any]:
    """Fit configured transformations on the training split only."""
    by_group = config.get("transformation", "by_group", default={})
    artifact: dict[str, Any] = {"group_methods": by_group, "feature_transformers": {}}

    for group_name, method in by_group.items():
        if method != "yeo_johnson":
            continue
        for column in feature_groups.get(group_name, []):
            if column not in train_frame.columns:
                continue
            transformer = PowerTransformer(method="yeo-johnson", standardize=False)
            transformer.fit(train_frame[[column]])
            artifact["feature_transformers"][column] = transformer
    return artifact


def apply_transformations(
    frame: pd.DataFrame,
    feature_groups: dict[str, list[str]],
    transformation_artifact: dict[str, Any],
) -> pd.DataFrame:
    """Apply configured transformations to a split."""
    transformed = frame.copy()
    by_group = transformation_artifact["group_methods"]
    feature_transformers = transformation_artifact["feature_transformers"]

    for group_name, method in by_group.items():
        columns = [column for column in feature_groups.get(group_name, []) if column in transformed]
        if method == "none":
            continue
        if method == "log1p":
            for column in columns:
                transformed[column] = np.log1p(transformed[column])
            continue
        if method == "yeo_johnson":
            for column in columns:
                transformer = feature_transformers.get(column)
                if transformer is None:
                    continue
                transformed[column] = transformer.transform(transformed[[column]]).ravel()
    return transformed

