"""Winsorization fit/apply utilities."""

from __future__ import annotations

from typing import Any

import pandas as pd

from core.config import PipelineConfig


def fit_winsorization_bounds(
    train_frame: pd.DataFrame,
    feature_groups: dict[str, list[str]],
    config: PipelineConfig,
) -> dict[str, dict[str, float]]:
    """Fit per-feature winsorization bounds using training data only."""
    bounds: dict[str, dict[str, float]] = {}
    default_config = config.get("outlier_treatment", "winsorization", "default")
    group_config = config.get("outlier_treatment", "winsorization", "by_group", default={})
    physiological_limits = config.get("validation", "physiological_limits", default={})

    for group_name, columns in feature_groups.items():
        if group_name == "identifier":
            continue
        quantiles = group_config.get(group_name, default_config)
        lower_q = float(quantiles["lower_quantile"])
        upper_q = float(quantiles["upper_quantile"])
        for column in columns:
            if column not in train_frame.columns:
                continue
            lower = float(train_frame[column].quantile(lower_q))
            upper = float(train_frame[column].quantile(upper_q))
            if column in physiological_limits:
                lower = max(lower, float(physiological_limits[column]["min"]))
                upper = min(upper, float(physiological_limits[column]["max"]))
            bounds[column] = {"lower": lower, "upper": upper}
    return bounds


def apply_winsorization(
    frame: pd.DataFrame,
    bounds: dict[str, dict[str, float]],
) -> pd.DataFrame:
    """Clip values to fitted winsorization bounds."""
    transformed = frame.copy()
    for column, limit in bounds.items():
        if column in transformed.columns:
            transformed[column] = transformed[column].clip(
                lower=limit["lower"],
                upper=limit["upper"],
            )
    return transformed

