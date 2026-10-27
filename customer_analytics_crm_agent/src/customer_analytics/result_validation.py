"""Sanity checks for outputs before interpretation/export."""
from __future__ import annotations
from typing import Any
from numbers import Real
import math
import numpy as np
import pandas as pd
from .analytics import AnalysisResult


def _nonfinite_paths(value: Any, path: str) -> list[str]:
    """Return paths to non-finite numeric values, including nested result metadata."""
    issues: list[str] = []
    if isinstance(value, (Real, np.number)) and not isinstance(value, (bool, np.bool_)):
        try:
            if not math.isfinite(float(value)):
                issues.append(path)
        except (TypeError, ValueError, OverflowError):
            issues.append(path)
    elif isinstance(value, dict):
        for key, nested in value.items():
            issues.extend(_nonfinite_paths(nested, f"{path}.{key}"))
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            issues.extend(_nonfinite_paths(nested, f"{path}[{index}]"))
    return issues


def validate_result(result: AnalysisResult) -> list[str]:
    """Return issues found; an empty list means the basic output checks passed."""
    issues: list[str] = []
    for path in _nonfinite_paths(result.summary, "summary"):
        issues.append(f"Numeric output {path!r} is not finite.")
    if result.table is not None:
        if result.table.columns.duplicated().any():
            issues.append("Result table contains duplicate column names.")
        for column in result.table.select_dtypes(include="number").columns:
            series = result.table[column].dropna()
            try:
                finite_mask = np.isfinite(series.to_numpy(dtype=float))
            except (TypeError, ValueError, OverflowError):
                issues.append(f"Numeric output column {column!r} could not be validated for finite values.")
                continue
            if not bool(finite_mask.all()):
                issues.append(f"Numeric output column {column!r} contains infinite or non-finite values.")
    return issues


def result_to_record(result: AnalysisResult) -> dict[str, Any]:
    """Serialize summary metadata without embedding the full result table."""
    return {
        "analysis_id": result.analysis_id,
        "name": result.name,
        "summary": result.summary,
        "row_count": len(result.table) if result.table is not None else 0,
        "columns": list(result.table.columns) if result.table is not None else [],
        "warnings": result.warnings or [],
    }
