"""Deterministic report export helpers, isolated from Streamlit for testability."""
from __future__ import annotations
import io
import json
import re
from typing import Any

import pandas as pd


def excel_bundle(
    original: pd.DataFrame,
    cleaned: pd.DataFrame,
    missing: pd.DataFrame,
    outliers: pd.DataFrame,
    feasibility_df: pd.DataFrame,
    results: dict[str, dict[str, Any]],
    metadata: dict[str, Any],
    *,
    duplicate_rows: pd.DataFrame | None = None,
    outlier_records: pd.DataFrame | None = None,
    audit: list[str] | None = None,
    interpretations: dict[str, dict[str, Any]] | None = None,
) -> bytes:
    """Build an Excel audit workbook from result objects with primary/auxiliary tables."""
    buff = io.BytesIO()
    used_sheets: set[str] = set()

    def write_df(writer, requested_name: str, frame: pd.DataFrame) -> None:
        base = re.sub(r"[\[\]:*?/\\]", "_", str(requested_name)).strip("'") or "Sheet"
        base = base[:31]
        name = base
        suffix = 1
        while name.casefold() in used_sheets:
            marker = f"_{suffix}"
            name = f"{base[:31-len(marker)]}{marker}"
            suffix += 1
        used_sheets.add(name.casefold())
        frame.head(200000).to_excel(writer, sheet_name=name, index=False)

    with pd.ExcelWriter(buff, engine="openpyxl") as writer:
        write_df(writer, "Original Preview", original)
        write_df(writer, "Analysis Data", cleaned)
        write_df(writer, "Missingness & Types", missing)
        write_df(writer, "Outlier Flags", outliers)
        write_df(writer, "Outlier Row Audit", outlier_records if outlier_records is not None else pd.DataFrame())
        write_df(writer, "Exact Duplicate Audit", duplicate_rows if duplicate_rows is not None else pd.DataFrame())
        write_df(writer, "Cleaning Audit", pd.DataFrame({"Audit entry": audit or ["No cleaning operations recorded."]}))

        dictionary = []
        for col in original.columns:
            series = original[col]
            examples = series.dropna().astype(str).head(3).tolist()
            dictionary.append({
                "Column": str(col), "Inferred dtype": str(series.dtype),
                "Non-missing values": int(series.notna().sum()), "Missing values": int(series.isna().sum()),
                "Unique values": int(series.nunique(dropna=True)), "Example values": " | ".join(examples),
                "Business meaning": "Confirm with the data owner; inferred type/alias is not authoritative business meaning.",
            })
        write_df(writer, "Data Dictionary", pd.DataFrame(dictionary))
        write_df(writer, "Feasibility", feasibility_df)
        write_df(writer, "Results Summary", pd.DataFrame([
            {"Analysis": key, "Summary": val.get("summary", ""), "Method": val.get("method", ""), "Sample size": val.get("sample_size", ""), "Note": val.get("note", "")}
            for key, val in results.items()
        ]))

        interpretation_rows = []
        for name, item in (interpretations or {}).items():
            interpretation_rows.append({
                "Analysis": name,
                "Overview": item.get("overview", ""),
                "Key findings": " | ".join(map(str, item.get("key_findings", []))),
                "Business implications": item.get("business_implications", ""),
                "Root-cause assessment": item.get("root_cause_assessment", ""),
                "Recommendations": " | ".join(map(str, item.get("recommendations", []))),
                "KPIs": " | ".join(map(str, item.get("kpis", []))),
                "Limitations": " | ".join(map(str, item.get("limitations", []))),
            })
        write_df(writer, "Interpretations", pd.DataFrame(interpretation_rows))

        for i, (name, result) in enumerate(results.items(), start=1):
            table = result.get("table")
            if isinstance(table, pd.DataFrame) and not table.empty:
                write_df(writer, f"R{i:02d}_{name}", table)
            for j, (aux_name, aux_table) in enumerate(result.get("auxiliary_tables", {}).items(), start=1):
                if isinstance(aux_table, pd.DataFrame) and not aux_table.empty:
                    write_df(writer, f"A{i:02d}_{j:02d}_{aux_name}", aux_table)
        write_df(writer, "Metadata", pd.DataFrame([
            {"Key": str(k), "Value": json.dumps(v, default=str) if isinstance(v, (dict, list, tuple)) else str(v)}
            for k, v in metadata.items()
        ]))
    return buff.getvalue()
