"""Evidence-bound markdown and CSV export helpers."""
from __future__ import annotations
from datetime import datetime, timezone
from typing import Any
import json
import pandas as pd
from .analytics import AnalysisResult


def build_markdown_report(
    filename: str,
    table_name: str,
    profile: dict[str, Any],
    feasibility: list[dict[str, Any]],
    results: list[AnalysisResult],
) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    status_counts: dict[str, int] = {}
    for item in feasibility:
        status = item["feasibility_status"]
        status_counts[status] = status_counts.get(status, 0) + 1
    lines = [
        "# Customer Analytics Report", "",
        f"- **Source file:** {filename}", f"- **Selected table/sheet:** {table_name}",
        f"- **Generated:** {now}", f"- **Rows:** {profile['row_count']:,}",
        f"- **Columns:** {profile['column_count']:,}", "",
        "## Executive summary", "",
        f"The selected table contains {profile['row_count']:,} rows and {profile['column_count']:,} columns. "
        f"The feasibility assessment evaluated {len(feasibility)} registered analyses; "
        f"{len(results)} supported analysis/analyses were executed in this MVP session.", "",
        "## Data-quality overview", "",
        f"- Exact duplicate rows flagged: {profile['exact_duplicate_rows']:,}",
        f"- Fully empty rows: {profile['fully_empty_rows']:,}",
        f"- Fully empty columns: {len(profile['fully_empty_columns'])}",
    ]
    for warning in profile.get("warnings", []):
        lines.append(f"- **Warning:** {warning}")
    lines.extend(["", "### Column profile", "", "| Column | Type | Missing | Missing % | Distinct |", "|---|---|---:|---:|---:|"])
    for column in profile["columns"]:
        lines.append(
            f"| {column['column']} | {column['inferred_kind']} | {column['missing_count']} | "
            f"{column['missing_pct']:.2f}% | {column['distinct_non_null']} |"
        )
    lines.extend(["", "## Feasibility summary", "", "| Analysis ID | Analysis | Status | Missing fields | Blockers / warnings |", "|---|---|---|---|---|"])
    for item in feasibility:
        reason = (item.get("blockers") or item.get("warnings") or "None recorded").replace("|", "/")
        lines.append(f"| {item['analysis_id']} | {item['analysis']} | {item['feasibility_status']} | {item['missing_fields'] or '—'} | {reason} |")
    lines.extend(["", "## Executed analyses", ""])
    if not results:
        lines.append("No supported analysis was executed. Review feasibility blockers and the MVP-supported analysis list.")
    for result in results:
        lines.extend([f"### {result.name} ({result.analysis_id})", "", "**Computed summary**", "", "```json", json.dumps(result.summary, indent=2, default=str), "```", ""])
        for warning in (result.warnings or []):
            lines.append(f"- **Caution:** {warning}")
        lines.append("")
        if result.table is not None:
            lines.append(f"Result table contains {len(result.table):,} rows and {len(result.table.columns):,} columns; download the CSV for full results.")
            lines.append("")
    lines.extend([
        "## Limitations", "",
        "- This MVP implements only a subset of the 35 registered analyses; other analyses are feasibility-assessed but not yet executed by code.",
        "- Feasibility classifications are rule-based and require validation against varied real-world schemas.",
        "- No LLM-generated narrative is used in this starter; summaries are generated from computed outputs to avoid unsupported numerical claims.",
        "- Results do not establish causation unless an appropriate causal design was used.",
        "- Review the mapping, record granularity, exclusions, business definitions and data-quality warnings before making decisions.",
        "",
    ])
    return "\n".join(lines)


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")
