"""Dataset profiling and quality diagnostics without automatic destructive cleaning."""
from __future__ import annotations
from typing import Any
import pandas as pd


def _infer_kind(series: pd.Series, column_name: str = "") -> str:
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    label = str(column_name).strip().lower()
    if any(token in label for token in ("date", "time", "timestamp", "created_at", "resolved_at", "closed_at")):
        return "datetime candidate"
    non_null = series.dropna()
    if len(non_null) and non_null.nunique() <= min(30, max(2, int(len(non_null) * 0.05))):
        return "categorical/text"
    return "text or unknown"


def profile_dataframe(df: pd.DataFrame) -> dict[str, Any]:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("profile_dataframe expects a pandas DataFrame")
    rows, cols = df.shape
    column_records: list[dict[str, Any]] = []
    for position, column in enumerate(df.columns):
        # Positional access remains safe when a malformed input has duplicate headers.
        series = df.iloc[:, position]
        missing = int(series.isna().sum())
        non_null = series.dropna()
        record: dict[str, Any] = {
            "column": str(column),
            "pandas_dtype": str(series.dtype),
            "inferred_kind": _infer_kind(series, str(column)),
            "missing_count": missing,
            "missing_pct": round(100 * missing / rows, 2) if rows else 0.0,
            "distinct_non_null": int(non_null.nunique()),
        }
        if record["inferred_kind"] in {"datetime", "datetime candidate"}:
            parsed_dates = pd.to_datetime(series, errors="coerce", utc=True)
            valid_dates = parsed_dates.dropna()
            record["date_parseable_count"] = int(valid_dates.size)
            record["date_parse_failure_count"] = int(series.notna().sum() - valid_dates.size)
            if not valid_dates.empty:
                record["date_min_utc"] = str(valid_dates.min())
                record["date_max_utc"] = str(valid_dates.max())
        if pd.api.types.is_numeric_dtype(series):
            numeric = pd.to_numeric(series, errors="coerce").dropna()
            if not numeric.empty:
                record.update({
                    "min": float(numeric.min()),
                    "median": float(numeric.median()),
                    "mean": float(numeric.mean()),
                    "max": float(numeric.max()),
                })
        column_records.append(record)

    exact_duplicate_rows = int(df.duplicated().sum()) if rows else 0
    fully_empty_rows = int(df.isna().all(axis=1).sum()) if rows and cols else 0
    fully_empty_columns = [str(df.columns[pos]) for pos in range(cols) if df.iloc[:, pos].isna().all()]
    return {
        "row_count": int(rows),
        "column_count": int(cols),
        "exact_duplicate_rows": exact_duplicate_rows,
        "fully_empty_rows": fully_empty_rows,
        "fully_empty_columns": fully_empty_columns,
        "columns": column_records,
        "memory_usage_bytes": int(df.memory_usage(index=True, deep=True).sum()),
        "warnings": _basic_warnings(df),
    }


def _basic_warnings(df: pd.DataFrame) -> list[str]:
    warnings: list[str] = []
    if df.empty:
        warnings.append("The selected table has no records.")
    if df.columns.duplicated().any():
        warnings.append("Duplicate column names were detected; resolve before analysis.")
    if df.shape[1] == 0:
        warnings.append("The selected table has no columns.")
    if int(df.duplicated().sum()) > 0:
        warnings.append("Exact duplicate rows exist; inspect record granularity before any removal.")
    return warnings
