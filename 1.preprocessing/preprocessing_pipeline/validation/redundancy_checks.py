"""Correlation and VIF checks on processed features."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from core.config import PipelineConfig


def run_redundancy_checks(
    train_frame: pd.DataFrame,
    config: PipelineConfig,
) -> dict[str, Any]:
    """Compute correlation and VIF summaries on the training split."""
    subject_id_column = config.get("dataset", "subject_id_column")
    threshold = float(config.get("validation", "correlation_threshold"))
    vif_threshold = float(config.get("validation", "vif_threshold"))

    feature_frame = train_frame.drop(columns=[subject_id_column])
    correlation_summary = _correlation_summary(feature_frame, threshold)
    vif_summary = _vif_summary(feature_frame, vif_threshold)
    return {
        "correlation": correlation_summary,
        "vif": vif_summary,
    }


def _correlation_summary(frame: pd.DataFrame, threshold: float) -> dict[str, Any]:
    corr_matrix = frame.corr().abs()
    pairs: list[dict[str, Any]] = []
    columns = corr_matrix.columns.tolist()
    for i, left in enumerate(columns):
        for right in columns[i + 1 :]:
            value = float(corr_matrix.loc[left, right])
            if value >= threshold:
                pairs.append({"feature_a": left, "feature_b": right, "correlation": value})
    pairs.sort(key=lambda item: item["correlation"], reverse=True)
    return {
        "threshold": threshold,
        "high_correlation_pair_count": len(pairs),
        "top_pairs": pairs[:20],
    }


def _vif_summary(frame: pd.DataFrame, threshold: float) -> dict[str, Any]:
    values = frame.to_numpy(dtype=float)
    features = frame.columns.tolist()
    vif_rows: list[dict[str, Any]] = []

    for index, feature in enumerate(features):
        target = values[:, index]
        predictors = np.delete(values, index, axis=1)
        if predictors.shape[1] == 0:
            vif = 1.0
        else:
            model = LinearRegression()
            model.fit(predictors, target)
            r_squared = float(model.score(predictors, target))
            if r_squared >= 0.999999:
                vif = float("inf")
            else:
                vif = float(1.0 / (1.0 - r_squared))
        vif_rows.append({"feature": feature, "vif": vif})

    flagged = [
        row
        for row in sorted(vif_rows, key=lambda item: (np.isfinite(item["vif"]), item["vif"]), reverse=True)
        if row["vif"] >= threshold or not np.isfinite(row["vif"])
    ]
    return {
        "threshold": threshold,
        "high_vif_feature_count": len(flagged),
        "top_features": flagged[:20],
    }

