"""Evidence-aware CRM capability lookup, comparison and gap assessment."""
from __future__ import annotations
import json
import math
import re
from pathlib import Path
from typing import Any

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "crm_capabilities.json"

FEATURES = [
    ("sales_pipeline", "Sales pipeline / deal management", "Sales management", ["pipeline", "deal", "opportunity", "sales stage"]),
    ("workflow_automation", "Workflow automation", "Automation", ["workflow", "automation", "trigger", "task", "notification"]),
    ("sales_analytics", "Sales reporting and dashboards", "Analytics", ["dashboard", "report", "insight", "forecast view", "sales analytics"]),
    ("predictive_forecasting", "Predictive sales forecasting / scoring", "Predictive intelligence", ["predict", "forecast", "einstein", "lead score", "scoring"]),
    ("health_churn", "Customer health / churn prediction", "Retention intelligence", ["churn", "retention", "customer health", "risk score", "inactive"]),
    ("marketing_automation", "Email / lifecycle marketing automation", "Marketing", ["email", "campaign", "marketing", "predictive sending", "journey"]),
    ("segmentation", "Customer segmentation and audience targeting", "Marketing / analytics", ["segment", "audience", "targeting", "personalization"]),
    ("customer_support", "Customer support / ticket workflows", "Service", ["ticket", "support", "service", "case management", "help desk"]),
    ("satisfaction_measurement", "Customer satisfaction / feedback measurement", "Customer experience", ["csat", "nps", "satisfaction", "feedback", "survey"]),
    ("apis_integrations", "APIs, webhooks and integrations", "Administration / extensibility", ["api", "webhook", "integration", "developer"]),
    ("custom_data_model", "Custom fields / data model", "Administration", ["custom field", "custom object", "data model", "schema"]),
    ("data_quality", "Duplicate management and data quality", "Data governance", ["duplicate", "deduplication", "data quality", "dedupe"]),
    ("ai_reporting", "AI-generated or natural-language analysis", "AI / analytics", ["ai report", "natural language", "ai-generated", "ai assistant", "ai agent"]),
    ("retention_actions", "Risk-based retention task routing", "Retention operations", ["retention task", "risk-based", "churn score", "task routing", "customer save"]),
]


def load_knowledge_base(path: str | Path | None = None) -> dict[str, Any]:
    source = Path(path) if path else DATA_PATH
    data = json.loads(source.read_text(encoding="utf-8"))
    return data


def validate_knowledge_base(data: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    platforms = {p.get("platform") for p in data.get("platforms", [])}
    ids: set[str] = set()
    for row in data.get("capabilities", []):
        for key in ("record_id", "platform", "capability_name", "category", "availability_status", "evidence_url", "evidence_note", "verification_status", "evidence_confidence"):
            if key not in row or row[key] in (None, ""):
                errors.append(f"{row.get('record_id','<unknown>')}: missing {key}")
        rid = row.get("record_id")
        if rid in ids:
            errors.append(f"Duplicate record_id: {rid}")
        ids.add(rid)
        if row.get("platform") not in platforms:
            errors.append(f"{rid}: platform not declared")
        if row.get("availability_status") not in {
            "verified_available", "available_plan_restrictions", "available_addon",
            "available_integration", "requires_configuration_or_custom_development",
            "not_verified", "verified_unavailable_within_scope"
        }:
            errors.append(f"{rid}: unsupported availability_status")
        if row.get("verification_status") == "verified":
            if not row.get("verification_date"):
                errors.append(f"{rid}: verified record must have verification_date")
            if not row.get("evidence_url", "").startswith("https://"):
                errors.append(f"{rid}: verified record must contain HTTPS evidence URL")
        if row.get("verification_status") != "verified" and row.get("availability_status") == "verified_unavailable_within_scope":
            errors.append(f"{rid}: unavailable claim cannot be unverified")
    return errors


def search_capabilities(query: str, platforms: list[str] | None = None, *, data: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    data = data or load_knowledge_base()
    q = re.sub(r"[^a-z0-9 ]+", " ", query.casefold()).strip()
    terms = {t for t in q.split() if len(t) > 2}
    synonym_map = {
        "churn": {"churn", "health", "retention", "risk", "inactive"},
        "retention": {"retention", "churn", "health", "risk"},
        "automation": {"automation", "workflow", "task", "trigger"},
        "reporting": {"report", "dashboard", "insight", "forecast"},
        "report": {"report", "dashboard", "insight", "analytics"},
        "forecast": {"forecast", "predictive", "scoring", "einstein"},
        "ai": {"ai", "natural", "language", "agent", "report"},
        "marketing": {"marketing", "email", "campaign", "journey", "sending"},
        "integration": {"integration", "api", "webhook", "developer"},
        "data": {"data", "quality", "duplicate", "custom", "field"},
        "customer": {"customer", "health", "churn", "satisfaction", "feedback", "retention"},
    }
    expanded = set(terms)
    for t in terms:
        expanded |= synonym_map.get(t, set())
    results = []
    for row in data.get("capabilities", []):
        if platforms and row.get("platform") not in platforms:
            continue
        text = " ".join(str(row.get(k, "")) for k in ("capability_name", "category", "description", "keywords", "evidence_note")).casefold()
        norm = re.sub(r"[^a-z0-9 ]+", " ", text)
        row_terms = set(norm.split())
        overlap = expanded & row_terms
        score = len(overlap)
        if q and q in norm:
            score += 5
        if score:
            enriched = dict(row)
            enriched["relevance_score"] = score
            results.append(enriched)
    return sorted(results, key=lambda r: (r["relevance_score"], r.get("verification_status") == "verified"), reverse=True)


def compare_capabilities(platforms: list[str], feature_ids: list[str], *, data: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    data = data or load_knowledge_base()
    rows = data.get("capabilities", [])
    output = []
    for feature_id in feature_ids:
        feature = next((f for f in FEATURES if f[0] == feature_id), None)
        if not feature:
            continue
        for platform in platforms:
            matches = [r for r in rows if r.get("platform") == platform and r.get("feature_id") == feature_id]
            rec = matches[0] if matches else None
            if rec:
                output.append({"feature_id": feature_id, "capability": feature[1], "category": feature[2], "platform": platform,
                               "status": rec.get("availability_status", "not_verified"), "verification_status": rec.get("verification_status", "needs_verification"),
                               "plan_scope": rec.get("plan_scope", "Not established"), "implementation_type": rec.get("implementation_type", "Not established"),
                               "limitations": rec.get("limitations", "Not established"), "evidence_url": rec.get("evidence_url", ""),
                               "evidence_note": rec.get("evidence_note", "No evidence note"), "verification_date": rec.get("verification_date", "")})
            else:
                output.append({"feature_id": feature_id, "capability": feature[1], "category": feature[2], "platform": platform,
                               "status": "not_verified", "verification_status": "needs_verification", "plan_scope": "Not researched in this matrix",
                               "implementation_type": "Unknown", "limitations": "No feature-specific source is recorded; this is not evidence of unavailability.",
                               "evidence_url": "", "evidence_note": "Knowledge-base coverage gap. Verify using official documentation and the selected plan.", "verification_date": ""})
    return output


def assess_gap(payload: dict[str, Any]) -> dict[str, Any]:
    """Conservative gap classification; feature absence requires scoped verified evidence."""
    verified = bool(payload.get("evidence_verified"))
    status = payload.get("availability_status", "not_verified")
    config = payload.get("configuration_status", "unknown")
    data_ready = payload.get("data_available", True)
    integration_needed = bool(payload.get("integration_needed", False))
    issue = payload.get("process_issue", False)
    if not verified or status == "not_verified":
        kind, finding = "Evidence/research gap", "The relevant product/plan capability needs verification; do not call it missing."
        confirmed = False
    elif data_ready is False:
        kind, finding = "Data quality gap", "The workflow lacks one or more required or trustworthy data inputs."
        confirmed = True
    elif integration_needed or status == "available_integration":
        kind, finding = "Integration gap", "The desired workflow depends on a cross-system connection or data flow."
        confirmed = False if config == "unknown" else True
    elif config == "unavailable_reported":
        kind, finding = "Subscription/cost constraint", "The team reports that the feature is unavailable in its current plan, but the exact subscription entitlement must be verified before confirming a limitation."
        confirmed = False
    elif status == "available_plan_restrictions" and config not in {"configured", "configured_ineffective"} or config == "plan_unknown":
        kind, finding = "Subscription/cost constraint", "Confirm the subscribed edition, limits, add-ons and seat entitlements before making a cost or availability claim."
        confirmed = False
    elif config == "not_configured" and status == "verified_available":
        kind, finding = "Configuration/adoption gap", "The feature is documented, but user-supplied configuration evidence says it is not configured or adopted."
        confirmed = True
    elif status == "verified_unavailable_within_scope" and config == "unavailable_confirmed":
        kind, finding = "Functional capability gap", "Official evidence supports unavailability only within the documented plan/product scope; consider workaround or alternative tooling."
        confirmed = True
    elif issue:
        kind, finding = "Process/adoption gap", "The workflow or ownership model may be insufficient even though product capability exists."
        confirmed = False
    else:
        kind, finding = "Analytics/decision-support gap", "The reported outcome need is not yet shown to be solved by the current configuration; validate through a workflow test."
        confirmed = False
    # Transparent weighted score. These are stakeholder ratings, not measured financial impact.
    weights = payload.get("weights", {"impact": 0.30, "frequency": 0.15, "manual_effort": 0.15, "customer_financial_consequence": 0.15, "differentiation": 0.10, "feasibility": 0.10, "customer_analytics_relevance": 0.05})
    dimensions = {k: int(payload.get(k, 3)) for k in weights}
    if any(v < 1 or v > 5 for v in dimensions.values()):
        raise ValueError("Prioritization ratings must be from 1 to 5.")
    weight_values = [float(v) for v in weights.values()]
    if any(not math.isfinite(v) or v < 0 for v in weight_values):
        raise ValueError("Prioritization weights must be finite, non-negative numbers.")
    total_w = sum(weight_values)
    if total_w <= 0:
        raise ValueError("At least one prioritization weight must be greater than zero.")
    score = round(sum(dimensions[k] * float(w) for k, w in weights.items()) / total_w / 5 * 100)
    confidence = int(payload.get("evidence_confidence", 1))
    if not verified:
        confidence_label = "Low — evidence not verified"
    elif confidence >= 4:
        confidence_label = "High documentation confidence"
    elif confidence >= 3:
        confidence_label = "Moderate documentation confidence"
    else:
        confidence_label = "Low documentation confidence"
    action_catalog = {
        "Evidence/research gap": {
            "owner": "CRM administrator / RevOps researcher",
            "complexity": "Low for verification; solution complexity unknown until evidence is reviewed",
            "actions": ["Open the cited official documentation and record the specific plan/edition and limits.", "Inspect the actual CRM tenant and test the relevant workflow with non-sensitive sample records.", "Update the evidence register; do not describe this as a product deficiency until the scope is verified."],
            "automation": "No automation change until entitlement and configuration are verified.",
            "integration": "Unknown; assess only after checking native functionality and data availability.",
            "benefits": "Reduces false-negative research claims and prevents unnecessary custom development or add-on purchase.",
            "kpis": ["Share of required capabilities with verified evidence", "Number of unresolved plan/configuration questions", "Time to complete capability verification"],
            "risks": ["Official documentation can change or be account/region specific.", "A product feature description is not proof that the customer's tenant has it enabled."]
        },
        "Data quality gap": {
            "owner": "Data steward / Analytics lead / CRM administrator",
            "complexity": "Medium; depends on source systems and identity resolution",
            "actions": ["Specify required fields and acceptable completeness/validity rules.", "Reconcile IDs, event dates, currency, row grain and business definitions before modelling.", "Re-run the same metric on the validated analysis population and retain a change log."],
            "automation": "Add validation checks and exception queues; route uncertain records for review rather than silently changing them.",
            "integration": "Potentially needed if required fields are held in separate systems.",
            "benefits": "Improves metric reliability and reduces manual reconciliation; no quantified saving is assumed.",
            "kpis": ["Required-field completeness", "Duplicate business-key rate", "Invalid date/value rate", "Share of records passing validation"],
            "risks": ["Incorrect merges can combine different customers.", "Valid outliers or refunds may be mistakenly removed."]
        },
        "Integration gap": {
            "owner": "IT / Integration owner / RevOps",
            "complexity": "Medium to High; confirm API access, ownership and data volume",
            "actions": ["Draw the source-to-target data flow and define record matching keys.", "Check official API/webhook documentation, authentication scopes, rate limits and cost.", "Pilot a read-only sync and validate freshness, duplicate handling and failure recovery."],
            "automation": "Use monitored, idempotent syncs and retry/dead-letter handling only after security approval.",
            "integration": "Likely; select an approved native connector or API/webhook implementation after discovery.",
            "benefits": "May reduce manual transfers and provide a more complete customer view.",
            "kpis": ["Sync success rate", "Data freshness lag", "Unmatched record rate", "Manual transfer time baseline vs pilot"],
            "risks": ["Credential exposure, rate limits, schema drift and privacy constraints.", "A demo interface is not a live integration."]
        },
        "Subscription/cost constraint": {
            "owner": "CRM administrator / Procurement / Functional owner",
            "complexity": "Low for entitlement check; financial impact requires quote and usage context",
            "actions": ["Confirm current product, edition, region, seats, add-ons, usage quotas and renewal terms.", "Compare configuring the existing plan, buying an add-on, integrating a tool or building a workflow.", "Estimate total cost of ownership using actual vendor quotes and internal effort."],
            "automation": "Do not build a workaround before confirming whether the capability is already licensed.",
            "integration": "Consider only after native entitlement and configuration are checked.",
            "benefits": "Avoids paying for duplicate functionality and supports a defensible buy/configure/build choice.",
            "kpis": ["Incremental subscription cost", "Feature adoption rate", "Implementation and maintenance effort", "Cost per completed workflow"],
            "risks": ["Plan names and entitlements change; undocumented assumptions can invalidate a comparison.", "Vendor list price may differ from contracted pricing."]
        },
        "Configuration/adoption gap": {
            "owner": "CRM administrator / Workflow owner / Frontline team lead",
            "complexity": "Low to Medium if the documented feature is licensed and supported",
            "actions": ["Document trigger conditions, record scope, action owner and exception paths.", "Configure in a sandbox or controlled pilot, test edge cases, and obtain approval before activation.", "Train users and audit whether the workflow is adopted after launch."],
            "automation": "Configure proposed task routing/notifications with explicit human review before any customer-facing action.",
            "integration": "Only if missing context is stored outside the CRM.",
            "benefits": "May reduce repetitive manual follow-up and improve process consistency.",
            "kpis": ["Workflow execution success rate", "Time from qualifying event to assigned task", "Task completion rate", "User adoption rate"],
            "risks": ["Over-triggering, missing permissions, duplicate tasks and poor ownership can disrupt operations."]
        },
        "Functional capability gap": {
            "owner": "Business owner / CRM product owner / IT architect",
            "complexity": "High until a documented product/plan limitation and requirements are confirmed",
            "actions": ["Define the minimum acceptance criteria and verify the product/plan scope with official evidence.", "Evaluate an approved integration, custom extension or alternative platform against cost, risk and maintainability.", "Pilot the smallest viable workflow before committing to a build."],
            "automation": "Use a human-approved action queue until a reliable and authorized execution path exists.",
            "integration": "Potentially needed; confirm with vendor docs and security/architecture review.",
            "benefits": "Could close a documented workflow requirement if the tested alternative meets acceptance criteria.",
            "kpis": ["Requirement coverage", "Cycle time for the target process", "Manual intervention rate", "Total cost of ownership"],
            "risks": ["Custom maintenance burden, integration failure, vendor lock-in and privacy/security exposure."]
        },
        "Process/adoption gap": {
            "owner": "Process owner / Sales, Marketing, Service or Customer Success leader",
            "complexity": "Medium; depends on role clarity and change management",
            "actions": ["Map the current workflow, hand-offs, decision rights and exception cases.", "Define a RACI, response SLA and escalation procedure.", "Pilot the revised process and compare the same KPI against a pre-intervention baseline."],
            "automation": "Automate only repetitive steps with stable business rules; retain an owner for exceptions.",
            "integration": "Not presumed; map actual hand-offs first.",
            "benefits": "May improve consistent execution even when CRM features already exist.",
            "kpis": ["Process cycle time", "Follow-up completion", "Exception rate", "Outcome KPI for the relevant workflow"],
            "risks": ["Low adoption, unclear accountability and inconsistent manager enforcement."]
        },
        "Analytics/decision-support gap": {
            "owner": "Customer Analytics lead / RevOps / Functional decision owner",
            "complexity": "Medium for a governed reporting/pilot layer; production prediction requires separate validation",
            "actions": ["Specify the decision, target user, frequency, threshold and next action the analysis must support.", "Compare native CRM reports/health scores with the required analytical definition and dataset.", "Test the proposed decision workflow with a baseline and human-reviewed recommendations."],
            "automation": "Generate a prioritized, human-approved queue; do not auto-send messages or alter CRM records without explicit integration and authorization.",
            "integration": "Could connect exported data or a documented API; no live integration is included in this prototype.",
            "benefits": "May translate existing metrics into more consistent prioritization and measurable follow-up.",
            "kpis": ["Time from signal to reviewed action", "Recommendation acceptance rate", "Eligible customer coverage", "Outcome KPI such as retention or conversion with a valid comparison"],
            "risks": ["Existing native features may already meet the need; labels can be biased by missing data; recommendations may be ignored."]
        }
    }
    plan = action_catalog[kind]
    return {"gap_type": kind, "description": finding, "confirmed_finding": confirmed,
            "priority_score": score, "priority": "High" if score >= 75 else "Medium" if score >= 50 else "Lower",
            "evidence_confidence": confidence_label, "assessed_on": __import__("datetime").date.today().isoformat(),
            "affected_stakeholder": plan["owner"], "root_cause_hypothesis": finding,
            "proposed_improvement": plan["actions"], "recommended_automation": plan["automation"],
            "recommended_integration": plan["integration"], "implementation_complexity": plan["complexity"],
            "expected_business_benefits": plan["benefits"], "kpis_to_monitor": plan["kpis"],
            "risks_and_limitations": plan["risks"],
            "measurement_plan": "Record a pre-intervention baseline using the same population, metric definition and time window. After the pilot, compare a like-for-like period or a suitable holdout; disclose seasonality and other changes. No impact percentage is assumed.",
            "recommended_next_step": "Validate the actual plan, configuration, data mapping and workflow against the linked source before implementation." if not confirmed else "Document the approved corrective action, owner, baseline KPI and follow-up review date."}
