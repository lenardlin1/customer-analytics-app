"""Versioned starter registry for the 35 customer analytics in the requirements."""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Iterable

REGISTRY_VERSION = "0.1.0"

@dataclass(frozen=True)
class AnalysisDefinition:
    analysis_id: str
    name: str
    required_fields: tuple[str, ...]
    description: str
    method: str
    category: str
    clarify_if_fields_present: tuple[str, ...] = ()
    notes: str = ""

    def as_dict(self) -> dict:
        item = asdict(self)
        item["required_fields"] = ", ".join(self.required_fields)
        item["clarify_if_fields_present"] = ", ".join(self.clarify_if_fields_present)
        return item

# These are baseline semantic requirements. Method-specific prerequisites and
# sample-size policies are added as the registry is tested and refined.
_DEFS = [
("CA-001","Customer Satisfaction Score (CSAT)",("satisfaction_score",),"Calculate satisfaction using a declared question and scoring convention.","deterministic metric","survey","satisfaction_definition"),
("CA-002","Churn analysis and prediction",("customer_id",),"Analyze defined churn outcomes; prediction requires eligible history and validation.","descriptive analysis; optional classifier","lifecycle","churn_definition"),
("CA-003","Customer retention analysis",("customer_id","order_date"),"Measure retention under a defined qualifying activity and observation window.","cohort/period analysis","lifecycle","retention_definition"),
("CA-004","Customer Lifetime Value (CLV/CLTV)",("customer_id","order_date","revenue"),"Historical customer value; predictive CLV is a separate validated task.","historical aggregation; optional prediction","value","value_definition"),
("CA-005","Customer journey analysis",("customer_id","event_time","event_type"),"Reconstruct time-ordered events and common journeys.","sequence/path analysis","journey","identity_definition"),
("CA-006","Customer segmentation",("customer_id",),"Build profiles or segments using meaningful customer-level features.","rule-based segmentation/clustering","segmentation","segment_objective"),
("CA-007","Feedback and sentiment analysis",("feedback_text",),"Analyze available text feedback and recurring themes.","text classification/theme analysis","experience"),
("CA-008","Customer behaviour and purchase patterns",("customer_id","order_date"),"Analyze observed purchase and activity patterns.","descriptive/time analysis","behaviour"),
("CA-009","Customer loyalty and repeat-purchase analysis",("customer_id","order_date"),"Analyze repeat purchasing; loyalty-program claims need loyalty evidence.","repeat-purchase metrics","loyalty","repeat_definition"),
("CA-010","Customer Effort Score (CES)",("effort_score",),"Calculate CES with declared question, scale and direction.","deterministic metric","survey","ces_definition"),
("CA-011","Net Promoter Score (NPS)",("nps_score",),"Calculate standard NPS from 0–10 recommendation responses.","deterministic metric","survey","nps_definition"),
("CA-012","Customer Acquisition Cost (CAC)",("campaign_cost",),"Calculate CAC only with compatible attributable new-customer counts and cost scope.","cost/attribution calculation","acquisition","cac_definition"),
("CA-013","CLV-to-CAC ratio",("customer_id","order_date","revenue","campaign_cost","new_customer_count"),"Compare compatible CLV and CAC definitions, cohorts, currency and periods.","derived ratio","value","clv_cac_definition"),
("CA-014","First Contact Resolution (FCR)",("ticket_id","first_contact_resolution"),"Calculate FCR only with a reliable issue/resolution and repeat-contact definition.","service metric","service","fcr_definition"),
("CA-015","Ticket response and resolution times",("ticket_id","created_at","resolved_at"),"Calculate elapsed or business-hour durations under a declared convention.","duration analysis","service","time_definition"),
("CA-016","Customer price sensitivity",("price","quantity"),"Analyze price-demand relationships when suitable independent price variation exists.","elasticity/regression/experiment","pricing","price_sensitivity_design"),
("CA-017","RFM analysis",("customer_id","order_date","revenue"),"Calculate recency, frequency and monetary value under a defined reference date.","customer aggregation/scoring","segmentation","rfm_definition"),
("CA-018","Revenue, sales and customer profitability",("revenue",),"Aggregate sales; profitability requires a defensible cost or margin measure.","aggregation/margin analysis","value","profitability_definition"),
("CA-019","Average Order Value (AOV) and purchase frequency",("order_id","revenue"),"Calculate order value and frequency with validated order/line granularity.","transaction aggregation","behaviour","order_granularity"),
("CA-020","Repeat-purchase rate and time between purchases",("customer_id","order_date"),"Measure repeat purchasing in a defined eligible population and window.","repeat-purchase metrics","lifecycle","repeat_definition"),
("CA-021","Cohort retention and customer lifecycle",("customer_id","order_date"),"Build comparable cohorts with defined entry and qualifying activity.","cohort analysis","lifecycle","cohort_definition"),
("CA-022","Customer funnel conversion and drop-off",("customer_id","stage"),"Measure stage movement with defined order and conversion window.","funnel analysis","journey","funnel_definition"),
("CA-023","Product affinity and market basket analysis",("order_id","product_id"),"Discover co-purchase rules from valid order baskets.","association rules","product"),
("CA-024","Cross-selling and upselling opportunities",("customer_id","product_id"),"Identify candidates using purchases and relevant product/eligibility rules.","rules/recommendation","product","recommendation_objective"),
("CA-025","Customer engagement and campaign performance",("campaign_id","conversion"),"Analyze engagement and conversions with valid exposure denominators.","campaign metrics","marketing","campaign_definition"),
("CA-026","Marketing attribution and campaign ROI",("campaign_id","conversion"),"Attribution needs touchpoint identity/time; ROI also needs cost and revenue.","attribution/ROI","marketing","attribution_definition"),
("CA-027","Service workload, SLA and ticket backlog",("ticket_id","created_at","ticket_status"),"Analyze workload and backlog; SLA compliance needs a target and status history.","service operations metrics","service","sla_definition"),
("CA-028","Complaint analysis and root-cause exploration",("feedback_text",),"Find themes and associations; causality requires additional design/evidence.","text/theme/association analysis","experience"),
("CA-029","Customer segment profiling and segment profitability",("customer_id","segment","revenue"),"Profile segment economics; profitability requires costs/margins.","segment aggregation","segmentation","segment_definition"),
("CA-030","Revenue concentration and customer dependency",("customer_id","revenue"),"Measure revenue shares and concentration by customer over a defined period.","concentration metrics","value","period_definition"),
("CA-031","Discount effectiveness and promotional sensitivity",("discount","revenue"),"Compare promotion outcomes; causal lift requires an appropriate comparison design.","comparative analysis/experiment","marketing","promotion_definition"),
("CA-032","Customer migration between lifecycle stages or segments",("customer_id","event_time","stage"),"Measure migration using timestamped stage/segment history.","transition analysis","lifecycle","stage_history_definition"),
("CA-033","Revenue forecasting and demand trends",("order_date","revenue"),"Forecast a defined series with suitable history and temporal validation.","time-series forecasting","forecast","forecast_horizon"),
("CA-034","Churn risk scoring and retention prioritization",("customer_id","cancellation_status"),"Requires a valid historical outcome, prediction-time features and validated model.","classification/decision analysis","lifecycle","churn_definition"),
("CA-035","Next-best-action and personalized recommendations",("customer_id",),"Recommend eligible actions from customer and action/product data.","business rules/recommendation","prescriptive","recommendation_objective"),
]

REGISTRY: tuple[AnalysisDefinition, ...] = tuple(
    AnalysisDefinition(
        analysis_id=a[0], name=a[1], required_fields=tuple(a[2]), description=a[3],
        method=a[4], category=a[5],
        clarify_if_fields_present=tuple(a[6].split(",")) if len(a) > 6 and a[6] else (),
        notes=a[7] if len(a) > 7 else "",
    ) for a in _DEFS
)


def registry_records() -> list[dict]:
    return [definition.as_dict() for definition in REGISTRY]


def get_analysis(analysis_id: str) -> AnalysisDefinition | None:
    return next((item for item in REGISTRY if item.analysis_id == analysis_id), None)
