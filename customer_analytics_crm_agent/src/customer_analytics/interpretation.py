"""Evidence-bound interpretation helpers. No LLM/provider is called in the starter."""
from __future__ import annotations
from typing import Any, Protocol
from .analytics import AnalysisResult


class NarrativeProvider(Protocol):
    """Optional future adapter for an LLM that writes prose from validated evidence.

    Implementations must not execute generated code or independently compute metrics.
    They should receive minimized, structured result records rather than raw customer
    rows, and should return narrative plus evidence references.
    """
    def explain(self, evidence: dict[str, Any]) -> dict[str, Any]: ...


def deterministic_interpretation(result: AnalysisResult) -> dict[str, Any]:
    """Return a transparent baseline interpretation based only on computed outputs."""
    summary = result.summary
    interpretation: list[str] = [f"Analysis: {result.name} ({result.analysis_id})."]
    for key, value in summary.items():
        interpretation.append(f"{key}: {value}.")
    limitations = list(result.warnings or [])
    if result.analysis_id == "CA-018":
        limitations.append("Historical customer revenue is not customer profit and is not predictive CLV.")
    if result.analysis_id == "CA-030":
        limitations.append("Revenue concentration describes historical dependency; it does not predict churn.")
    return {
        "analysis_id": result.analysis_id,
        "interpretation_lines": interpretation,
        "limitations": list(dict.fromkeys(limitations)),
        "narrative_provider_used": False,
    }
