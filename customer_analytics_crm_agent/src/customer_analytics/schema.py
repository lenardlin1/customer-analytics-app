"""Conservative semantic column mapping. The original column labels are preserved."""
from __future__ import annotations
import re
from typing import Any
import pandas as pd

ALIASES: dict[str, tuple[str, ...]] = {
    "customer_id": ("customer_id", "customerid", "customer key", "cust_id", "custcode", "client_id", "account_id", "user_id", "member_id"),
    "order_id": ("order_id", "orderid", "transaction_id", "invoice_id", "purchase_id", "basket_id", "receipt_id"),
    "order_date": ("order_date", "transaction_date", "purchase_date", "invoice_date", "order_timestamp", "purchase_timestamp"),
    "revenue": ("revenue", "order_value", "order_amount", "transaction_value", "sales_amount", "net_sales", "total_amount", "monetary_value", "line_revenue"),
    "quantity": ("quantity", "qty", "units_sold", "item_quantity"),
    "product_id": ("product_id", "productid", "sku", "item_id", "stock_code"),
    "product_category": ("product_category", "category", "product_group", "item_category"),
    "price": ("unit_price", "selling_price", "price", "item_price", "list_price"),
    "unit_cost": ("unit_cost", "cost_per_unit", "product_cost"),
    "cost": ("cost", "total_cost", "cost_amount", "acquisition_cost"),
    "profit": ("profit", "gross_profit", "contribution_profit", "margin_amount"),
    "discount": ("discount", "discount_amount", "discount_percent", "discount_rate", "promotion_discount"),
    "segment": ("segment", "customer_segment", "customer_group", "rfm_segment"),
    "signup_date": ("signup_date", "sign_up_date", "registration_date", "customer_since", "created_customer_at"),
    "cancellation_status": ("churned", "churn_status", "cancelled", "canceled", "cancellation_status", "subscription_cancelled"),
    "nps_score": ("nps", "nps_score", "recommendation_score", "likelihood_to_recommend", "recommend_score"),
    "satisfaction_score": ("csat", "csat_score", "satisfaction_score", "customer_satisfaction"),
    "effort_score": ("ces", "ces_score", "effort_score", "customer_effort_score"),
    "survey_question": ("survey_question", "question", "question_text", "item_text"),
    "survey_response": ("survey_response", "response", "answer", "rating_response"),
    "feedback_text": ("feedback_text", "comment", "comments", "review_text", "complaint_text", "message_text", "customer_feedback"),
    "campaign_id": ("campaign_id", "campaign", "campaign_name", "marketing_campaign"),
    "campaign_cost": ("campaign_cost", "marketing_cost", "ad_spend", "advertising_spend", "campaign_spend", "acquisition_spend"),
    "new_customer_count": ("new_customer_count", "new_customers", "acquired_customers", "customer_acquisitions"),
    "channel": ("channel", "marketing_channel", "acquisition_channel", "source", "medium"),
    "conversion": ("conversion", "converted", "conversion_flag", "converted_flag", "is_converted"),
    "event_time": ("event_time", "event_timestamp", "timestamp", "event_date", "activity_timestamp", "interaction_time"),
    "event_type": ("event_type", "event_name", "activity_type", "interaction_type", "event"),
    "stage": ("stage", "funnel_stage", "lifecycle_stage", "pipeline_stage", "journey_stage"),
    "ticket_id": ("ticket_id", "case_id", "support_ticket_id", "service_request_id"),
    "ticket_status": ("ticket_status", "case_status", "support_status", "status"),
    "first_contact_resolution": ("first_contact_resolution", "fcr", "resolved_first_contact", "first_contact_flag"),
    "issue_id": ("issue_id", "contact_reason_id", "case_group_id", "problem_id"),
    "created_at": ("created_at", "ticket_created_at", "case_created_at", "opened_at", "created_time"),
    "first_response_at": ("first_response_at", "first_reply_at", "first_response_time", "responded_at"),
    "resolved_at": ("resolved_at", "closed_at", "resolution_timestamp", "ticket_resolved_at"),
    "sla_target": ("sla_target", "sla_due_at", "sla_hours", "response_sla"),
    "priority": ("priority", "ticket_priority", "case_priority"),
}


def normalize_label(value: Any) -> str:
    """Normalize a header for matching; never changes the source dataframe."""
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def suggest_column_mapping(df: pd.DataFrame) -> dict[str, str | None]:
    """Map canonical fields only when an alias match is unique and unambiguous."""
    normalized_columns: dict[str, list[str]] = {}
    for column in df.columns:
        normalized_columns.setdefault(normalize_label(column), []).append(str(column))

    result: dict[str, str | None] = {}
    for canonical, aliases in ALIASES.items():
        possible: list[str] = []
        for alias in aliases:
            key = normalize_label(alias)
            matches = normalized_columns.get(key, [])
            if len(matches) == 1 and matches[0] not in possible:
                possible.append(matches[0])
        result[canonical] = possible[0] if len(possible) == 1 else None
    return result


def validate_mapping(df: pd.DataFrame, mapping: dict[str, str | None]) -> list[str]:
    """Return mapping errors, including accidentally reusing one column for multiple roles."""
    errors: list[str] = []
    for canonical, actual in mapping.items():
        if actual is not None and actual not in df.columns:
            errors.append(f"{canonical!r} maps to missing column {actual!r}.")
    mapped = [value for value in mapping.values() if value is not None]
    duplicates = sorted({value for value in mapped if mapped.count(value) > 1})
    for column in duplicates:
        errors.append(f"Column {column!r} is mapped to multiple semantic fields; confirm whether this is intentional.")
    return errors
