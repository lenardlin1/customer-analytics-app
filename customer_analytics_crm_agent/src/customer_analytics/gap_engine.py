"""Rule-based CRM gap classification and transparent priority scoring."""
from __future__ import annotations
from datetime import date
from urllib.parse import urlparse
from typing import Any

GAP_TYPES = {
    "configuration": "Capability exists but is not configured or used correctly.",
    "data": "Required data is missing, incomplete, inconsistent, or not linked.",
    "plan_limitation": "Capability may require a different plan, licence, or add-on.",
    "integration": "Required data or functionality is in another system.",
    "analytics": "Available reporting does not meet the required analytical depth.",
    "process": "Workflow, ownership, or operating process needs redesign.",
    "governance": "Privacy, permissions, auditability, or data-quality controls need attention.",
    "unverified": "Evidence is insufficient to confirm the capability or gap."
}

def validate_capability_record(record: dict[str, Any]) -> list[str]:
    """Return validation errors; do not treat a URL alone as proof."""
    errors = []
    required = ["record_id", "platform", "capability_name", "availability_status",
                "plan_dependency", "evidence_url", "evidence_note",
                "verification_status", "last_updated"]
    for key in required:
        if not str(record.get(key, "") or "").strip():
            errors.append(f"Missing required field: {key}")
    url = str(record.get("evidence_url", "") or "")
    parsed = urlparse(url)
    if url and (parsed.scheme != "https" or not parsed.netloc):
        errors.append("evidence_url must be a valid HTTPS URL")
    if record.get("verification_status") == "verified":
        if not record.get("verified_on"):
            errors.append("verified_on is required when verification_status is 'verified'")
        if not str(record.get("evidence_note", "")).strip():
            errors.append("A supporting evidence_note is required for verified records")
    if record.get("availability_status") not in {
        "available", "not_available", "plan_dependent", "integration_required",
        "custom_build", "unverified"
    }:
        errors.append("availability_status is not a supported value")
    if record.get("verification_status") not in {"verified", "partially_verified", "needs_verification"}:
        errors.append("verification_status is not a supported value")
    return errors

def classify_gap(
    *,
    evidence_verified: bool,
    availability_status: str,
    plan_dependency: str = "unknown",
    data_available: bool = True,
    configuration_checked: bool = False,
    integration_needed: bool = False,
    analytics_depth_insufficient: bool = False,
    process_issue: bool = False,
    governance_issue: bool = False,
) -> tuple[str, str]:
    """Classify a suspected gap conservatively; unverified evidence wins."""
    if governance_issue:
        return "governance", GAP_TYPES["governance"]
    if not evidence_verified or availability_status == "unverified":
        return "unverified", GAP_TYPES["unverified"]
    if not data_available:
        return "data", GAP_TYPES["data"]
    if plan_dependency in {"higher_plan", "paid_addon", "separate_license"} or availability_status == "plan_dependent":
        return "plan_limitation", GAP_TYPES["plan_limitation"]
    if integration_needed or availability_status == "integration_required":
        return "integration", GAP_TYPES["integration"]
    if configuration_checked and availability_status == "available":
        return "configuration", GAP_TYPES["configuration"]
    if analytics_depth_insufficient or availability_status == "custom_build":
        return "analytics", GAP_TYPES["analytics"]
    if process_issue:
        return "process", GAP_TYPES["process"]
    if availability_status == "not_available":
        return "analytics", "Capability is documented as unavailable for the assessed scope; compare custom-build and alternative-tool options."
    return "unverified", GAP_TYPES["unverified"]

def priority_score(impact: int, urgency: int, frequency: int,
                   evidence_confidence: int, feasibility: int) -> int:
    """Weighted 0-100 project heuristic. Each input must be an integer from 1 to 5."""
    values = [impact, urgency, frequency, evidence_confidence, feasibility]
    if any(not isinstance(v, int) or isinstance(v, bool) or v < 1 or v > 5 for v in values):
        raise ValueError("All score inputs must be integers from 1 to 5.")
    score = 100 * (
        0.30 * impact / 5 +
        0.20 * urgency / 5 +
        0.15 * frequency / 5 +
        0.20 * evidence_confidence / 5 +
        0.15 * feasibility / 5
    )
    return round(score)

def priority_band(score: int, critical_risk: bool = False) -> str:
    if critical_risk:
        return "Critical review required"
    if score >= 75:
        return "High"
    if score >= 50:
        return "Medium"
    return "Lower"

def assess_gap(payload: dict[str, Any]) -> dict[str, Any]:
    """Return a transparent gap assessment from user-supplied evidence and ratings."""
    gap_type, explanation = classify_gap(
        evidence_verified=payload.get("evidence_verified", False),
        availability_status=payload.get("availability_status", "unverified"),
        plan_dependency=payload.get("plan_dependency", "unknown"),
        data_available=payload.get("data_available", True),
        configuration_checked=payload.get("configuration_checked", False),
        integration_needed=payload.get("integration_needed", False),
        analytics_depth_insufficient=payload.get("analytics_depth_insufficient", False),
        process_issue=payload.get("process_issue", False),
        governance_issue=payload.get("governance_issue", False),
    )
    score = priority_score(
        payload["impact"], payload["urgency"], payload["frequency"],
        payload["evidence_confidence"], payload["feasibility"]
    )
    # A low confidence score prevents a result being represented as a confirmed finding.
    confidence = payload["evidence_confidence"]
    confirmed = bool(payload.get("evidence_verified", False)) and confidence >= 3 and gap_type != "unverified"
    return {
        "gap_type": gap_type,
        "classification_explanation": explanation,
        "priority_score": score,
        "priority": priority_band(score, payload.get("critical_risk", False)),
        "confirmed_finding": confirmed,
        "confirmation_note": (
            "Evidence and confidence threshold passed; retain source evidence for review."
            if confirmed else
            "Treat as a hypothesis or validation task, not a confirmed CRM defect."
        ),
        "assessed_on": date.today().isoformat()
    }
