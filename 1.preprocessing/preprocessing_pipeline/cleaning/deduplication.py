"""Duplicate row handling."""

from __future__ import annotations

import pandas as pd


def remove_exact_duplicates(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Remove rows that are exact duplicates across all columns."""
    duplicate_count = int(frame.duplicated().sum())
    return frame.drop_duplicates().reset_index(drop=True), duplicate_count

