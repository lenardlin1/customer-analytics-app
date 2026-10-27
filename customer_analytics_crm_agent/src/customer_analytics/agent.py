"""Orchestrator for the controlled analytical workflow; no generated-code execution."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import pandas as pd
from .analytics import AnalysisResult, AnalysisError, run_analysis, SUPPORTED_ANALYSES
from .feasibility import assess_feasibility
from .profiling import profile_dataframe
from .schema import suggest_column_mapping
from .result_validation import validate_result

@dataclass
class DatasetAssessment:
    profile: dict[str, Any]
    suggested_mapping: dict[str, str | None]
    feasibility: list[dict[str, Any]]


def assess_dataset(df: pd.DataFrame, confirmed_definitions: set[str] | None = None) -> DatasetAssessment:
    profile = profile_dataframe(df)
    mapping = suggest_column_mapping(df)
    feasibility = assess_feasibility(df, mapping, confirmed_definitions=confirmed_definitions)
    return DatasetAssessment(profile, mapping, feasibility)


def execute_approved_analyses(
    df: pd.DataFrame,
    mapping: dict[str, str | None],
    feasibility: list[dict[str, Any]],
    approved_analysis_ids: list[str],
) -> tuple[list[AnalysisResult], list[dict[str, str]]]:
    """Run only allowlisted analyses explicitly supplied by the approved plan."""
    by_id = {item["analysis_id"]: item for item in feasibility}
    results: list[AnalysisResult] = []
    errors: list[dict[str, str]] = []
    for analysis_id in approved_analysis_ids:
        item = by_id.get(analysis_id)
        if item is None:
            errors.append({"analysis_id": analysis_id, "error": "Analysis is not present in the current feasibility report."})
            continue
        if item["feasibility_status"] not in {"Feasible", "Partially feasible"}:
            errors.append({"analysis_id": analysis_id, "error": f"Execution blocked because status is {item['feasibility_status']}. Resolve blockers first."})
            continue
        if analysis_id not in SUPPORTED_ANALYSES:
            errors.append({"analysis_id": analysis_id, "error": "Registered but not implemented in this MVP. No calculation was attempted."})
            continue
        try:
            result = run_analysis(analysis_id, df, mapping)
            result_issues = validate_result(result)
            if result_issues:
                errors.append({"analysis_id": analysis_id, "error": "Result validation failed: " + "; ".join(result_issues)})
                continue
            results.append(result)
        except (AnalysisError, ValueError, TypeError, KeyError) as exc:
            errors.append({"analysis_id": analysis_id, "error": str(exc)})
    return results, errors
