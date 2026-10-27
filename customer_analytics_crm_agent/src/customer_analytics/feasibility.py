"""Rule-based feasibility assessment; not an LLM-generated guess."""
from __future__ import annotations
from typing import Any
import warnings as pywarnings
import pandas as pd
from .registry import REGISTRY
from .schema import validate_mapping

VALID_STATUSES = {"Feasible", "Partially feasible", "Not feasible", "Needs clarification"}


def assess_feasibility(
    df: pd.DataFrame,
    mapping: dict[str, str | None],
    *,
    confirmed_definitions: set[str] | None = None,
    ambiguous_fields: set[str] | None = None,
) -> list[dict[str, Any]]:
    confirmed_definitions = confirmed_definitions or set()
    ambiguous_fields = ambiguous_fields or set()
    mapping_errors = validate_mapping(df, mapping)
    results: list[dict[str, Any]] = []
    for definition in REGISTRY:
        missing = [field for field in definition.required_fields if not mapping.get(field)]
        absent_columns = [
            f"{field} → {mapping.get(field)!r} (column not found)"
            for field in definition.required_fields
            if mapping.get(field) is not None and mapping.get(field) not in df.columns
        ]
        blockers: list[str] = []
        warnings: list[str] = []
        if mapping_errors:
            warnings.extend(mapping_errors)
        if absent_columns:
            blockers.extend(absent_columns)
        if df.empty or df.shape[1] == 0:
            blockers.append("The selected dataset has no usable rows or columns.")
            status = "Not feasible"
        elif missing or absent_columns:
            blockers.extend([f"Required field '{field}' was not mapped." for field in missing])
            status = "Not feasible"
        elif mapping_errors or any(field in ambiguous_fields for field in definition.required_fields):
            blockers.append("Resolve invalid/reused or ambiguous semantic column mappings before analysis.")
            status = "Needs clarification"
        else:
            required_actual = [mapping[field] for field in definition.required_fields]
            missing_rates = []
            for actual in required_actual:
                if actual in df.columns and len(df):
                    missing_rates.append(float(df[actual].isna().mean()))
            worst_missing = max(missing_rates, default=0.0)
            if worst_missing >= 0.20:
                status = "Partially feasible"
                warnings.append(
                    f"At least one required field has {worst_missing:.0%} missing values; "
                    "inspect exclusions and sensitivity before accepting results."
                )
            else:
                status = "Feasible"
                if worst_missing > 0:
                    warnings.append("Some required data is missing; calculations must report eligible counts.")
            required_clarifications = [
                key for key in definition.clarify_if_fields_present
                if key not in confirmed_definitions
            ]
            if required_clarifications and status == "Feasible":
                status = "Needs clarification"
                blockers.append(
                    "Confirm business definition(s): " + ", ".join(required_clarifications) + "."
                )

        if definition.analysis_id == "CA-002":
            if not mapping.get("cancellation_status") and not mapping.get("order_date"):
                status = "Not feasible"
                blockers.append("Churn analysis requires cancellation outcomes or dated customer activity from which an explicit inactivity rule can be defined.")
            elif not mapping.get("cancellation_status"):
                warnings.append("No explicit churn outcome is mapped; derive churn only after confirming an inactivity window and observation cutoff.")
                if status == "Feasible":
                    status = "Needs clarification"
        if definition.analysis_id == "CA-006":
            candidate_features = ("revenue", "order_date", "segment", "product_category", "channel", "satisfaction_score", "nps_score", "effort_score", "quantity")
            if not any(mapping.get(field) for field in candidate_features):
                status = "Not feasible"
                blockers.append("Customer ID is present, but no meaningful customer-level segmentation feature was mapped.")
        if definition.analysis_id == "CA-012":
            if not mapping.get("customer_id") and not mapping.get("new_customer_count"):
                status = "Not feasible"
                blockers.append("CAC requires an attributable new-customer count or customer identities that can establish eligible acquisitions.")
            else:
                warnings.append("Verify that the denominator counts new customers attributable to the same cost scope and period; all customer IDs are not automatically new acquisitions.")
                if status == "Feasible":
                    status = "Needs clarification"
        if definition.analysis_id == "CA-014" and not mapping.get("issue_id") and not mapping.get("first_contact_resolution"):
            status = "Not feasible"
            blockers.append("Reliable FCR requires a first-contact resolution indicator or issue/contact history that can establish repeat contact.")
        if definition.analysis_id == "CA-034":
            if not mapping.get("cancellation_status"):
                status = "Not feasible"
                blockers.append("Predictive churn requires a valid historical churn outcome or an outcome derived before model training.")
            else:
                candidate_features = ("revenue", "order_date", "stage", "satisfaction_score", "nps_score", "effort_score", "channel", "quantity", "product_category", "signup_date", "event_time")
                if not any(mapping.get(field) for field in candidate_features):
                    status = "Not feasible"
                    blockers.append("A churn label and customer ID are not enough; at least one meaningful prediction-time feature must be available.")
                else:
                    warnings.append("Feature availability at prediction time, target definition, event counts, leakage and holdout performance must still be validated before training.")
        if definition.analysis_id == "CA-013":
            if not mapping.get("new_customer_count"):
                status = "Not feasible"
                blockers.append("A compatible CAC denominator is not available; CLV-to-CAC cannot be calculated reliably.")
            else:
                warnings.append("Confirm matching periods, cohorts, currency and whether CLV is revenue- or margin-based.")
        if definition.analysis_id == "CA-027" and not mapping.get("sla_target"):
            warnings.append("SLA compliance cannot be calculated without an SLA target; ticket counts/status may still support workload reporting.")
            if status == "Feasible":
                status = "Partially feasible"
        if definition.analysis_id == "CA-004" and not mapping.get("unit_cost") and not mapping.get("cost") and not mapping.get("profit"):
            warnings.append("Only revenue-based historical value is potentially available; margin-based CLV needs suitable cost/margin data, and predictive CLV needs separate model validation.")
        if definition.analysis_id == "CA-016" and status in {"Feasible", "Partially feasible"}:
            warnings.append("Price sensitivity requires suitable price variation and controls or an experimental design; field presence alone is insufficient.")
        if definition.analysis_id == "CA-026" and (not mapping.get("event_time") or not mapping.get("customer_id")):
            warnings.append("Attribution needs linked, time-stamped customer touchpoints; campaign and conversion columns alone may only support campaign summaries.")
            if status == "Feasible":
                status = "Partially feasible"
        if definition.analysis_id == "CA-017" and mapping.get("order_id") is None and status == "Feasible":
            warnings.append("No order ID mapped. Frequency may default to row count only if one row represents one transaction.")
            status = "Partially feasible"
        if definition.analysis_id in {"CA-018", "CA-029"}:
            if not mapping.get("profit") and not mapping.get("unit_cost") and not mapping.get("cost"):
                warnings.append("Revenue/value analysis may be possible, but profitability requires a suitable cost or margin measure.")
                if status == "Feasible":
                    status = "Partially feasible"
        if definition.analysis_id == "CA-012" and not mapping.get("customer_id"):
            warnings.append("Acquisition cost alone is not CAC; attributable new-customer count and period are also required.")
            if status == "Feasible":
                status = "Partially feasible"

        # Validate parseability for required semantic fields. Header presence alone is
        # not enough when date/numeric values are stored as malformed text.
        numeric_fields = {
            "revenue", "quantity", "price", "unit_cost", "cost", "profit",
            "campaign_cost", "new_customer_count", "nps_score",
            "satisfaction_score", "effort_score",
        }
        date_fields = {"order_date", "event_time", "signup_date", "created_at", "first_response_at", "resolved_at"}
        for semantic in definition.required_fields:
            actual = mapping.get(semantic)
            if not actual or actual not in df.columns:
                continue
            series = df[actual]
            nonmissing = series.notna()
            if semantic in numeric_fields:
                numeric = pd.to_numeric(series, errors="coerce")
                parse_failures = int((nonmissing & numeric.isna()).sum())
                valid_numeric = int(numeric.notna().sum())
                if valid_numeric == 0:
                    if status != "Not feasible":
                        status = "Needs clarification"
                    blockers.append(
                        f"Field '{semantic}' has no parseable numeric values; confirm its meaning, scale, and number/currency format."
                    )
                elif parse_failures:
                    if status == "Feasible":
                        status = "Partially feasible"
                    warnings.append(
                        f"Field '{semantic}' contains {parse_failures} non-numeric value(s); affected rows must be handled explicitly."
                    )
            elif semantic in date_fields:
                # Parse failures are reported below; suppress pandas' heuristic-format
                # warning because the feasibility result itself captures this condition.
                with pywarnings.catch_warnings():
                    pywarnings.filterwarnings(
                        "ignore",
                        message="Could not infer format, so each element will be parsed individually.*",
                        category=UserWarning,
                    )
                    parsed = pd.to_datetime(series, errors="coerce", utc=True)
                parse_failures = int((nonmissing & parsed.isna()).sum())
                valid_dates = int(parsed.notna().sum())
                if valid_dates == 0:
                    if status != "Not feasible":
                        status = "Needs clarification"
                    blockers.append(
                        f"Field '{semantic}' has no parseable dates; confirm the date format or field mapping."
                    )
                elif parse_failures:
                    if status == "Feasible":
                        status = "Partially feasible"
                    warnings.append(
                        f"Field '{semantic}' contains {parse_failures} unparseable date value(s); affected rows must be handled explicitly."
                    )

        # Standard NPS only accepts numeric recommendation scores from 0 through 10.
        # Mixed valid/invalid values can be partially analyzed, but invalid scores
        # must be reported rather than silently included in the denominator.
        if definition.analysis_id == "CA-011" and mapping.get("nps_score") in df.columns:
            actual = mapping["nps_score"]
            raw_scores = df[actual]
            numeric_scores = pd.to_numeric(raw_scores, errors="coerce")
            valid_scores = numeric_scores.between(0, 10, inclusive="both")
            invalid_count = int((raw_scores.notna() & ~valid_scores).sum())
            if int(valid_scores.sum()) == 0:
                status = "Not feasible"
                blockers.append("No valid numeric NPS recommendation scores in the inclusive 0–10 range were found.")
            elif invalid_count:
                if status == "Feasible":
                    status = "Partially feasible"
                warnings.append(f"{invalid_count} NPS response(s) are non-numeric or outside 0–10 and must be excluded or reviewed explicitly.")

        results.append({
            "analysis_id": definition.analysis_id,
            "analysis": definition.name,
            "category": definition.category,
            "feasibility_status": status,
            "required_fields": ", ".join(definition.required_fields),
            "missing_fields": ", ".join(missing),
            "blockers": " ".join(blockers),
            "warnings": " ".join(dict.fromkeys(warnings)),
            "method": definition.method,
            "description": definition.description,
            "clarification_keys": ", ".join(definition.clarify_if_fields_present),
        })
    return results
