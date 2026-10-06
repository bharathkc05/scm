"""Feature grouping utilities driven by config."""

from __future__ import annotations

import re
from typing import Iterable


def _matches_any(column: str, patterns: Iterable[str]) -> bool:
    return any(re.match(pattern, column) for pattern in patterns)


def build_feature_groups(
    columns: list[str],
    feature_group_config: dict[str, dict[str, object]],
) -> dict[str, list[str]]:
    """Map features into configured groups."""
    groups: dict[str, list[str]] = {}
    for group_name, rules in feature_group_config.items():
        include = [column for column in rules.get("include", []) if column in columns]
        regex = rules.get("regex")
        matched = [column for column in columns if regex and re.match(str(regex), column)]
        groups[group_name] = sorted(set(include + matched), key=columns.index)
    return groups


def match_feature_patterns(columns: list[str], patterns: list[str]) -> list[str]:
    """Return columns matching any configured regular expression."""
    return [column for column in columns if _matches_any(column, patterns)]

