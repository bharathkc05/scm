"""Feature removal based on predetermined config rules."""

from __future__ import annotations

import re

import pandas as pd


def get_features_to_remove(
    columns: list[str],
    explicit_features: list[str],
    regex_patterns: list[str],
) -> list[str]:
    """Resolve features to remove from explicit names and regex patterns."""
    removed = {feature for feature in explicit_features if feature in columns}
    for pattern in regex_patterns:
        removed.update(column for column in columns if re.match(pattern, column))
    return sorted(removed, key=columns.index)


def remove_features(frame: pd.DataFrame, features_to_remove: list[str]) -> pd.DataFrame:
    """Drop predetermined features from a dataframe."""
    return frame.drop(columns=features_to_remove, errors="ignore").copy()

