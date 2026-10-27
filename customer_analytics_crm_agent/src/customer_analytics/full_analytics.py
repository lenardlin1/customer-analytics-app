from __future__ import annotations

import io
import json
import re
import hashlib
from datetime import date
from typing import Any

import numpy as np
import pandas as pd


APP_TITLE = "Customer Analytics Intelligence Agent"
NOT_MAPPED = "— Not mapped —"


# Field definitions: aliases are used only to suggest a mapping. The user can override them.
FIELD_DEFS = {
    "customer_id": ("Customer ID / customer key", ["customer id", "customer_id", "client id", "client_id", "account id", "contact id", "user id", "customer key"]),
    "order_id": ("Order / transaction ID", ["order id", "order_id", "transaction id", "transaction_id", "invoice id", "purchase id", "ticket order"]),
    "txn_date": ("Transaction / purchase date", ["transaction date", "purchase date", "order date", "invoice date", "date of purchase", "date", "created date"]),
    "amount": ("Transaction / revenue amount", ["amount", "revenue", "sales", "order value", "order amount", "transaction amount", "total spend", "net sales", "purchase value", "monetary"]),
    "quantity": ("Quantity purchased", ["quantity", "qty", "units", "items sold", "units sold"]),
    "unit_price": ("Unit / transaction price", ["unit price", "price", "selling price", "sale price", "transaction price", "item price"]),
    "event_type": ("Journey event / stage", ["event type", "event", "stage", "journey stage", "activity type", "touchpoint", "step", "status event"]),
    "event_date": ("Journey event timestamp", ["event date", "event timestamp", "activity date", "touchpoint date", "event time"]),
    "feedback_text": ("Customer feedback text", ["feedback", "comment", "review", "review text", "customer comment", "response text", "survey comment", "message", "sentiment text"]),
    "csat_score": ("CSAT / satisfaction score", ["csat", "csat score", "satisfaction score", "customer satisfaction", "satisfaction rating", "overall rating"]),
    "ces_score": ("CES / effort score", ["ces", "ces score", "customer effort score", "effort score", "ease score"]),
    "nps_score": ("NPS recommendation score (0–10)", ["nps", "nps score", "recommendation score", "likelihood to recommend", "promoter score"]),
    "churn_flag": ("Churn / cancellation flag", ["churned", "churn flag", "churn status", "cancelled", "canceled", "is churned", "churn"]),
    "ticket_id": ("Support ticket ID", ["ticket id", "case id", "support id", "incident id"]),
    "ticket_created": ("Ticket created timestamp", ["ticket created", "created at", "created time", "opened at", "case created", "ticket open date"]),
    "ticket_resolved": ("Ticket resolved timestamp", ["resolved at", "resolution date", "ticket resolved", "closed at", "closed date", "case closed"]),
    "first_response": ("First response timestamp or duration", ["first response time", "first response at", "first reply at", "response timestamp"]),
    "fcr_flag": ("First-contact resolution flag", ["fcr", "first contact resolution", "resolved first contact", "resolved on first contact", "one touch resolution"]),
    "acquisition_spend": ("Acquisition / marketing spend", ["acquisition spend", "marketing spend", "ad spend", "campaign cost", "acquisition cost", "sales and marketing cost", "spend"]),
    "new_customers_count": ("New customers acquired (count)", ["new customers", "new customer count", "customers acquired", "acquired customers", "new clients", "conversion count"]),
    "new_customer_flag": ("New-customer flag (row-level)", ["is new customer", "new customer flag", "acquired flag", "new customer", "first purchase flag"]),
    "channel": ("Acquisition / sales / marketing channel", ["channel", "acquisition channel", "source", "lead source", "campaign", "marketing channel", "traffic source"]),
    "product": ("Product / item", ["product", "product name", "sku", "item", "item name", "product id"]),
    "loyalty_flag": ("Loyalty-program / member flag", ["loyalty member", "loyalty status", "member status", "loyalty enrolled", "rewards member"]),
}

METRIC_NAMES = [
    "Customer Satisfaction (CSAT)", "Customer Churn", "Customer Retention", "Customer Lifetime Value (CLV)",
    "Customer Journey", "Customer Segmentation", "Feedback Sentiment", "Customer Behaviour", "Customer Loyalty",
    "Customer Effort Score (CES)", "Net Promoter Score (NPS)", "Customer Acquisition Cost (CAC)",
    "CLV:CAC and CAC:CLV ratios", "First Contact Resolution (FCR)", "Average Ticket / Response Time",
    "Price Sensitivity", "RFM Analysis",
]


def normalized_name(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).casefold()).strip()


def make_unique_columns(columns) -> list[str]:
    seen: dict[str, int] = {}
    out = []
    for col in columns:
        name = str(col).strip() or "Unnamed"
        if name not in seen:
            seen[name] = 0
            out.append(name)
        else:
            seen[name] += 1
            out.append(f"{name}.{seen[name]}")
    return out


def read_uploaded_file(uploaded, sheet_name: str | None = None) -> tuple[pd.DataFrame, list[str]]:
    data = uploaded.getvalue()
    name = uploaded.name.lower()
    if name.endswith(".csv"):
        try:
            df = pd.read_csv(io.BytesIO(data), low_memory=False)
        except UnicodeDecodeError:
            df = pd.read_csv(io.BytesIO(data), encoding="latin-1", low_memory=False)
        return df, []
    if name.endswith((".xlsx", ".xlsm")):
        book = pd.ExcelFile(io.BytesIO(data))
        sheets = book.sheet_names
        selected = sheet_name if sheet_name in sheets else sheets[0]
        return pd.read_excel(io.BytesIO(data), sheet_name=selected), sheets
    if name.endswith(".xls"):
        try:
            book = pd.ExcelFile(io.BytesIO(data), engine="xlrd")
            sheets = book.sheet_names
            selected = sheet_name if sheet_name in sheets else sheets[0]
            return pd.read_excel(io.BytesIO(data), sheet_name=selected, engine="xlrd"), sheets
        except Exception as exc:
            raise ValueError("Legacy .xls reading requires xlrd. Convert the workbook to .xlsx or install xlrd.") from exc
    raise ValueError("Please upload a CSV, XLSX, XLSM, or XLS file.")


def guess_column(df: pd.DataFrame, aliases: list[str]) -> str:
    cols = list(df.columns)
    norm_cols = {c: normalized_name(c) for c in cols}
    norm_aliases = [normalized_name(a) for a in aliases]
    # Prefer exact match; otherwise match an alias as a phrase inside the column name.
    for alias in norm_aliases:
        for col, norm_col in norm_cols.items():
            if norm_col == alias:
                return col
    for alias in norm_aliases:
        for col, norm_col in norm_cols.items():
            if alias and alias in norm_col:
                return col
    return NOT_MAPPED


def parse_dates(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", format="mixed")


def numeric(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    cleaned = series.astype(str).str.replace(r"[,$₹€£%\s]", "", regex=True)
    cleaned = cleaned.replace({"nan": np.nan, "None": np.nan, "": np.nan, "-": np.nan})
    return pd.to_numeric(cleaned, errors="coerce")


def map_value(df: pd.DataFrame, mapping: dict[str, str], key: str) -> pd.Series | None:
    col = mapping.get(key, NOT_MAPPED)
    if col == NOT_MAPPED or col not in df.columns:
        return None
    return df[col]


def truthy_flag(series: pd.Series, positive_values: str = "yes,true,1,churned,cancelled,canceled,closed,lost,promoter,resolved") -> pd.Series:
    positives = {normalized_name(x) for x in positive_values.split(",") if x.strip()}
    values = series.astype(str).map(normalized_name)
    return values.isin(positives)


def quality_report(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    missing = pd.DataFrame({
        "column": df.columns,
        "dtype": [str(df[c].dtype) for c in df.columns],
        "missing_count": [int(df[c].isna().sum()) for c in df.columns],
        "missing_percent": [round(float(df[c].isna().mean() * 100), 2) for c in df.columns],
        "unique_values": [int(df[c].nunique(dropna=True)) for c in df.columns],
    }).sort_values(["missing_percent", "column"], ascending=[False, True])
    nums = df.select_dtypes(include=np.number)
    out_rows = []
    for col in nums.columns:
        s = pd.to_numeric(nums[col], errors="coerce").dropna()
        if len(s) < 4:
            out_rows.append({"column": col, "n": len(s), "iqr_low": np.nan, "iqr_high": np.nan, "flagged_outliers": 0, "flagged_percent": 0.0})
            continue
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        mask = (s < low) | (s > high)
        out_rows.append({"column": col, "n": len(s), "iqr_low": round(float(low), 4), "iqr_high": round(float(high), 4), "flagged_outliers": int(mask.sum()), "flagged_percent": round(float(mask.mean() * 100), 2)})
    return missing, pd.DataFrame(out_rows)


def outlier_record_report(df: pd.DataFrame, treatment: str = "Flag only (recommended)") -> pd.DataFrame:
    """Return row-level numeric IQR flags with a proposed/actioned treatment. Flags are not proof of error."""
    excluded = {"id", "phone", "postal", "zip", "nps", "csat", "ces", "rating", "score", "flag", "count", "year", "month", "day"}
    numeric_cols = [c for c in df.select_dtypes(include=np.number).columns if not any(t in normalized_name(c).split() for t in excluded)]
    rows: list[dict[str, Any]] = []
    flagged_row_indices: set[Any] = set()
    for col in numeric_cols:
        s = pd.to_numeric(df[col], errors="coerce")
        valid = s.dropna()
        if len(valid) < 4:
            continue
        q1, q3 = valid.quantile([0.25, 0.75])
        iqr = q3 - q1
        low, high = float(q1 - 1.5 * iqr), float(q3 + 1.5 * iqr)
        mask = (s < low) | (s > high)
        for position, (idx, value) in enumerate(s.items()):
            if pd.isna(value) or not (value < low or value > high):
                continue
            flagged_row_indices.add(idx)
            rows.append({"Source_Row_Number": position + 2, "Column": str(col), "Observed_Value": df.at[idx, col],
                         "IQR_Lower_Bound": low, "IQR_Upper_Bound": high, "Detection_Method": "1.5×IQR rule",
                         "Potential_Issue": "Statistical outlier candidate; validate business context and source record.",
                         "Treatment_Selected": treatment, "Change_Applied": False})
    report = pd.DataFrame(rows)
    if report.empty:
        return pd.DataFrame(columns=["Source_Row_Number", "Column", "Observed_Value", "IQR_Lower_Bound", "IQR_Upper_Bound", "Detection_Method", "Potential_Issue", "Treatment_Selected", "Change_Applied"])
    if treatment in ("Cap to IQR bounds", "Replace flagged values with column median"):
        report["Change_Applied"] = True
    elif treatment == "Drop rows with any eligible IQR outlier":
        report["Change_Applied"] = True
    report["IQR_Lower_Bound"] = report["IQR_Lower_Bound"].round(4)
    report["IQR_Upper_Bound"] = report["IQR_Upper_Bound"].round(4)
    return report.sort_values(["Source_Row_Number", "Column"]).reset_index(drop=True)


def apply_cleaning(df: pd.DataFrame, remove_exact_duplicates: bool, outlier_method: str) -> tuple[pd.DataFrame, list[str]]:
    work = df.copy()
    audit = []
    if remove_exact_duplicates:
        before = len(work)
        work = work.drop_duplicates().copy()
        removed = before - len(work)
        audit.append(f"Removed {removed} exact duplicate row(s).")
    else:
        audit.append("Exact duplicate rows were retained; they are reported in the quality audit.")

    exclude_terms = ["id", "phone", "postal", "zip", "nps", "csat", "ces", "rating", "score", "flag", "count", "year", "month", "day"]
    numeric_cols = [c for c in work.select_dtypes(include=np.number).columns if not any(t in normalized_name(c).split() for t in exclude_terms)]
    if outlier_method == "Flag only (recommended)":
        audit.append("Outliers were flagged for review; no numeric values were modified.")
        return work, audit

    if not numeric_cols:
        audit.append("No eligible numeric columns were found for the selected outlier treatment.")
        return work, audit

    if outlier_method in ("Cap to IQR bounds", "Replace flagged values with column median"):
        changed = 0
        for col in numeric_cols:
            s = pd.to_numeric(work[col], errors="coerce")
            valid = s.dropna()
            if len(valid) < 4:
                continue
            q1, q3 = valid.quantile([0.25, 0.75])
            iqr = q3 - q1
            low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            mask = (s < low) | (s > high)
            changed += int(mask.sum())
            if outlier_method == "Cap to IQR bounds":
                work[col] = s.clip(lower=low, upper=high)
            else:
                work.loc[mask, col] = float(valid.median())
        audit.append(f"Applied '{outlier_method}' to {changed} flagged cell(s) across eligible numeric columns. Review the outlier report; genuine extreme values may have been altered.")
    elif outlier_method == "Drop rows with any eligible IQR outlier":
        combined = pd.Series(False, index=work.index)
        for col in numeric_cols:
            s = pd.to_numeric(work[col], errors="coerce")
            valid = s.dropna()
            if len(valid) < 4:
                continue
            q1, q3 = valid.quantile([0.25, 0.75])
            iqr = q3 - q1
            low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            combined |= (s < low) | (s > high)
        before = len(work)
        work = work.loc[~combined].copy()
        audit.append(f"Dropped {before - len(work)} row(s) with an IQR outlier in at least one eligible numeric column. This can remove valid customers or transactions; compare results with the unfiltered data.")
    return work, audit


def feasibility(mapping: dict[str, str], df: pd.DataFrame, extra: dict[str, Any]) -> pd.DataFrame:
    has = lambda *keys: all(mapping.get(k, NOT_MAPPED) != NOT_MAPPED for k in keys)
    id_date = has("customer_id", "txn_date")
    id_amount = has("customer_id", "amount")
    event_date = has("event_date") or has("txn_date")
    rows: list[dict[str, str]] = []
    defs = [
        ("Customer Satisfaction (CSAT)", has("csat_score"), "Needs a satisfaction score column.", "Use customer feedback text sentiment as a qualitative alternative; it is not equivalent to CSAT."),
        ("Customer Churn", has("churn_flag") or id_date, "Best method needs an explicit churn/cancellation label. Customer ID plus dates only supports an inactivity-based proxy.", "Report customers who have become inactive for a chosen period and label it a churn-risk proxy, not confirmed churn."),
        ("Customer Retention", id_date, "Needs a stable customer ID and transaction/activity date.", "With customer IDs only, report unique customers and repeat-record counts; a time-based retention rate needs dates."),
        ("Customer Lifetime Value (CLV)", id_amount, "Needs customer IDs and revenue/transaction values for observed historical customer value.", "Report historical revenue per customer if those fields exist; predictive CLV also needs a prediction horizon and defensible retention/margin assumptions."),
        ("Average Order Value (AOV)", has("order_id", "amount"), "Needs order IDs, revenue amounts, and confirmation of whether each row is an order or an order line.", "If order granularity is unclear, reconcile order totals first; report revenue totals without labeling them AOV."),
        ("Customer Journey", has("customer_id", "event_type") and event_date, "Needs customer IDs, event/stage labels and event timestamps.", "If there are only purchase records, analyse repeat purchase patterns or RFM rather than claiming to reconstruct a full journey."),
        ("Customer Segmentation", (len(df.select_dtypes(include=np.number).columns) >= 2) or id_amount or id_date, "Needs useful customer-level attributes; IDs alone cannot create meaningful segments.", "Produce descriptive groups from available fields or request purchase, recency, frequency, channel or demographic variables."),
        ("Feedback Sentiment", has("feedback_text"), "Needs a free-text comment, review or message column.", "Analyse structured ratings or category counts if available, but do not call them sentiment analysis."),
        ("Customer Behaviour", has("customer_id") and (has("txn_date") or has("amount") or has("event_type")), "Needs customer IDs and at least dates, amounts or activity/events.", "Summarise available purchase or activity attributes; obtain dates and IDs for longitudinal patterns."),
        ("Customer Loyalty", has("customer_id") and (has("txn_date") or has("loyalty_flag")), "Needs repeated customer activity or loyalty-program membership fields.", "Use repeat-purchase rate as a loyalty proxy if customer IDs and dates exist; loyalty is broader than repeat purchase."),
        ("Customer Effort Score (CES)", has("ces_score"), "Needs a CES/effort score column.", "Use service resolution time or customer feedback themes as a related diagnostic, not as a calculated CES."),
        ("Net Promoter Score (NPS)", has("nps_score"), "Needs a 0–10 recommendation score.", "Use CSAT or sentiment as a different satisfaction signal; neither can be converted reliably into NPS."),
        ("Customer Acquisition Cost (CAC)", (has("acquisition_spend", "new_customers_count") or (has("acquisition_spend", "customer_id", "new_customer_flag")) or (extra.get("manual_spend") is not None and extra.get("manual_new_customers") is not None)), "Needs acquisition spend and the count of newly acquired customers for the same period.", "If only spend or only new-customer count exists, report that component and request the missing value; do not invent CAC."),
        ("CLV:CAC and CAC:CLV ratios", (id_amount and (has("acquisition_spend", "new_customers_count") or (extra.get("manual_spend") is not None and extra.get("manual_new_customers") is not None))), "Needs a defensible customer-value estimate and CAC for comparable cohorts/periods.", "Report historical revenue per customer and CAC separately; mark any ratio as a proxy rather than predictive unit economics."),
        ("First Contact Resolution (FCR)", has("fcr_flag"), "Needs a first-contact-resolution flag or sufficient interaction history to derive it reliably.", "Report ticket resolution rate or response/resolution time if available; these are not FCR."),
        ("Average Ticket / Response Time", has("ticket_created", "ticket_resolved") or has("ticket_created", "first_response"), "Needs ticket creation and resolution timestamps, or creation and first-response timestamps/durations.", "Report whichever timestamps exist; otherwise request the appropriate service timestamps."),
        ("Price Sensitivity", has("unit_price", "quantity"), "Needs observed prices and quantities; causal sensitivity requires suitable price variation, comparable products/time periods and controls or an experiment.", "Report descriptive price/quantity association if available, or discount and sales mix trends; do not claim causality."),
        ("RFM Analysis", id_date and id_amount, "Needs customer ID, purchase date and monetary transaction value.", "If one field is missing, report the available dimensions separately and request the missing field; do not fabricate a full RFM score."),
    ]
    for name, available, reason, alternative in defs:
        rows.append({
            "Analysis": name,
            "Status": "Calculable (subject to field validation)" if available else "Not calculable from current mapping",
            "Reason / requirement": "Required columns appear to be mapped; the app will still validate data types and usable records." if available else reason,
            "Alternative": "Run the analysis and report limitations; avoid interpreting association as causation." if available else alternative,
        })
    # AOV must not silently treat line items as order totals or duplicate order IDs as unique orders.
    if has("order_id", "amount"):
        order_col = mapping["order_id"]
        duplicate_orders = int(df[order_col].dropna().duplicated(keep=False).sum())
        aov_row = next((r for r in rows if r["Analysis"] == "Average Order Value (AOV)"), None)
        grain = extra.get("row_granularity", "Mixed / unsure")
        if aov_row and grain == "Mixed / unsure":
            aov_row["Status"] = "Needs clarification"
            aov_row["Reason / requirement"] = "Order ID and amount columns are mapped, but row granularity has not been confirmed."
            aov_row["Alternative"] = "Confirm one row per order or one row per order line before calculating AOV."
        elif aov_row and duplicate_orders and grain == "One transaction / order per row":
            aov_row["Status"] = "Needs clarification"
            aov_row["Reason / requirement"] = f"{duplicate_orders} rows are involved in repeated order IDs; these may be duplicate records or line items."
            aov_row["Alternative"] = "Review the duplicate-order report and either remove verified exact duplicates or select order-line granularity if appropriate."
    # CES was intentionally deduplicated: the original request listed it twice.
    return pd.DataFrame(rows)


def clean_for_numeric(series: pd.Series) -> pd.Series:
    return numeric(series).replace([np.inf, -np.inf], np.nan)


def month_retention(df: pd.DataFrame, cust_col: str, date_col: str) -> pd.DataFrame:
    tmp = df[[cust_col, date_col]].copy()
    tmp[date_col] = parse_dates(tmp[date_col])
    tmp = tmp.dropna(subset=[cust_col, date_col]).copy()
    if tmp.empty:
        return pd.DataFrame()
    tmp["period"] = tmp[date_col].dt.to_period("M")
    first = tmp.groupby(cust_col)["period"].min().rename("cohort")
    tmp = tmp.join(first, on=cust_col)
    tmp["period_number"] = (tmp["period"].dt.year - tmp["cohort"].dt.year) * 12 + (tmp["period"].dt.month - tmp["cohort"].dt.month)
    cohort_counts = tmp.groupby(["cohort", "period_number"])[cust_col].nunique().unstack(fill_value=0)
    cohort_sizes = tmp.groupby("cohort")[cust_col].nunique()
    rate = cohort_counts.div(cohort_sizes, axis=0) * 100
    rate.index = rate.index.astype(str)
    rate.columns = [f"Month {int(c)}" for c in rate.columns]
    return rate.round(1).reset_index(names="Cohort month")


def compute_metrics(df: pd.DataFrame, mapping: dict[str, str], options: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    meta: dict[str, Any] = {"rows_analysed": int(len(df)), "columns": int(df.shape[1]), "analyses": {}}

    def add(name: str, summary: str, table: pd.DataFrame | None = None, note: str = ""):
        results[name] = {"summary": summary, "table": table, "note": note}
        meta["analyses"][name] = {"summary": summary, "note": note}

    def col(key):
        return map_value(df, mapping, key)

    # CSAT
    s = col("csat_score")
    if s is not None:
        x = clean_for_numeric(s).dropna()
        if len(x):
            threshold = float(options.get("csat_threshold", 4))
            satisfied = (x >= threshold).mean() * 100
            dist = x.value_counts().sort_index().rename_axis("Score").reset_index(name="Responses")
            add("Customer Satisfaction (CSAT)", f"Mean satisfaction score: {x.mean():.2f} (n={len(x)}). Share at or above the configured threshold ({threshold:g}): {satisfied:.1f}%.", dist, "The satisfied threshold is configurable; verify it matches your survey scale and organisation's definition.")
        else:
            add("Customer Satisfaction (CSAT)", "Mapped column contains no usable numeric satisfaction scores.", note="Check the scale, text labels or column mapping.")

    # Churn: explicit flag preferred, otherwise inactivity proxy.
    churn = col("churn_flag")
    if churn is not None:
        positives_text = options.get("positive_churn_labels", "yes,true,1,churned,cancelled,canceled,terminated,lost")
        vals = churn.dropna().astype(str)
        if len(vals):
            positive = truthy_flag(vals, positives_text)
            rate = float(positive.mean() * 100)
            add("Customer Churn", f"Observed positive churn-flag rate: {rate:.1f}% across {len(vals)} non-missing rows ({int(positive.sum())} positive flags).", pd.DataFrame({"Flag value": vals.value_counts().index, "Rows": vals.value_counts().values}), "The positive labels are configurable. Confirm that the selected labels mean churn for this dataset; 'closed' or 'lost' may be ambiguous.")
    elif col("customer_id") is not None and col("txn_date") is not None:
        c, d = col("customer_id"), col("txn_date")
        tmp = pd.DataFrame({"customer": c, "date": parse_dates(d)}).dropna()
        if len(tmp):
            cutoff = tmp["date"].max()
            days = int(options.get("inactivity_days", 90))
            last = tmp.groupby("customer")["date"].max()
            inactive = (cutoff - last).dt.days >= days
            add("Customer Churn (inactivity proxy)", f"{inactive.mean()*100:.1f}% of observed customers have no recorded activity in the last {days} days before {cutoff.date()} ({int(inactive.sum())}/{len(inactive)}). This is an inactivity proxy, not confirmed churn.", pd.DataFrame({"Customer ID": last.index.astype(str), "Last recorded activity": last.values, "Days inactive": (cutoff-last).dt.days.values, "Flagged inactive": inactive.values}).sort_values("Days inactive", ascending=False).head(100), "A dataset's last date is not necessarily the true business snapshot date. Validate the cutoff and inactivity window before interpreting churn.")

    # Retention
    if col("customer_id") is not None and col("txn_date") is not None:
        t = month_retention(df, mapping["customer_id"], mapping["txn_date"])
        if not t.empty:
            add("Customer Retention", f"Cohort retention matrix generated for {t.shape[0]} acquisition cohorts. Month 0 is 100% by construction.", t, "Retention is based on customers with recorded activity in each calendar month. Interpret carefully if event data are incomplete or the business is not expected to have monthly repeat activity.")
        else:
            add("Customer Retention", "No valid customer IDs and dates remained after validation.")

    # Average order value: only calculate if row granularity is established.
    order_col, amount_col = col("order_id"), col("amount")
    if order_col is not None and amount_col is not None:
        temp = pd.DataFrame({"order": order_col, "amount": clean_for_numeric(amount_col)}).dropna(subset=["order", "amount"])
        grain = options.get("row_granularity", "Mixed / unsure")
        if temp.empty:
            add("Average Order Value (AOV)", "AOV was not calculated because no valid order IDs and numeric amounts remained.", note="Check the order ID and amount mappings and missingness report.")
        elif grain == "One order line / item per row":
            order_values = temp.groupby("order", dropna=True)["amount"].sum()
            add("Average Order Value (AOV)", f"Average order value: {order_values.mean():,.2f} across {len(order_values)} distinct orders, after summing line amounts by order ID.", order_values.sort_values(ascending=False).head(100).rename_axis("Order ID").reset_index(name="Order total"), "This method assumes each row is a line/item amount, not an order total repeated across line items. Verify currency, returns and order completeness.")
        elif grain == "One transaction / order per row" and not temp["order"].duplicated().any():
            order_values = temp.groupby("order", dropna=True)["amount"].sum()
            add("Average Order Value (AOV)", f"Average order value: {order_values.mean():,.2f} across {len(order_values)} distinct orders.", order_values.sort_values(ascending=False).head(100).rename_axis("Order ID").reset_index(name="Order total"), "Assumes each row represents a complete order total. Confirm that order amounts are not duplicated in joins or item-level exports.")
        else:
            duplicate_order_count = int(temp.loc[temp["order"].duplicated(keep=False), "order"].nunique())
            add("Average Order Value (AOV)", "AOV was not calculated because order granularity is ambiguous or repeated order IDs were found in a declared order-level dataset.", temp[temp["order"].duplicated(keep=False)].groupby("order").size().rename("Rows per order").reset_index().head(100), f"{duplicate_order_count} distinct order ID(s) have repeated rows. Confirm whether these are legitimate order lines, duplicate transactions or repeated order totals before calculating AOV.")

    # Historical customer value / CLV proxy
    if col("customer_id") is not None and col("amount") is not None:
        cust_col, amt_col = mapping["customer_id"], mapping["amount"]
        temp = pd.DataFrame({"customer": df[cust_col], "amount": clean_for_numeric(df[amt_col])}).dropna()
        if len(temp):
            cust_rev = temp.groupby("customer")["amount"].sum().sort_values(ascending=False)
            add("Customer Lifetime Value (historical proxy)", f"Observed revenue per customer over the file's covered period: mean {cust_rev.mean():,.2f}, median {cust_rev.median():,.2f}; {len(cust_rev)} customers. This is historical observed value, not predictive lifetime value.", cust_rev.head(100).rename_axis("Customer ID").reset_index(name="Observed revenue"), "Currency and period come from the source data. Refunds/negative amounts are retained; determine whether the measure should be net revenue or contribution margin before using it as CLV.")

    # Journey
    cust, event, edate = col("customer_id"), col("event_type"), col("event_date")
    if edate is None:
        edate = col("txn_date")
    if cust is not None and event is not None and edate is not None:
        tmp = pd.DataFrame({"customer": cust, "event": event.astype(str), "date": parse_dates(edate)}).dropna(subset=["customer", "date"])
        if len(tmp):
            tmp = tmp.sort_values(["customer", "date"])
            seq = tmp.groupby("customer")["event"].apply(lambda z: " → ".join(z.astype(str).head(8))).value_counts().head(30).rename_axis("Observed event sequence (first 8 events/customer)").reset_index(name="Customers")
            add("Customer Journey", f"Summarised {len(tmp)} timestamped event records across {tmp['customer'].nunique()} customers. Top observed sequences are shown.", seq, "This describes recorded event order, not causal attribution. Missing touchpoints or timestamps can distort paths.")

    # Segmentation from numeric features, aggregated at customer level if ID exists.
    feature_df = df.select_dtypes(include=np.number).copy()
    feature_df = feature_df.loc[:, [c for c in feature_df.columns if not any(t in normalized_name(c).split() for t in ["id", "phone", "zip", "postal"])]]
    if col("customer_id") is not None and len(feature_df.columns) >= 2:
        tmp = df[[mapping["customer_id"]]].join(feature_df)
        tmp = tmp.rename(columns={mapping["customer_id"]: "customer"}).groupby("customer").mean(numeric_only=True)
    elif len(feature_df.columns) >= 2:
        tmp = feature_df.copy()
        tmp.index = [f"row_{i}" for i in range(len(tmp))]
    else:
        tmp = pd.DataFrame()
    if tmp.shape[0] >= 6 and tmp.shape[1] >= 2:
        try:
            from sklearn.cluster import KMeans
            from sklearn.impute import SimpleImputer
            from sklearn.preprocessing import StandardScaler
            k = min(max(2, int(options.get("n_clusters", 3))), 8, tmp.shape[0] - 1)
            imputed = SimpleImputer(strategy="median").fit_transform(tmp.replace([np.inf, -np.inf], np.nan))
            scaled = StandardScaler().fit_transform(imputed)
            labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(scaled)
            seg = tmp.copy()
            seg["Segment"] = [f"Segment {x+1}" for x in labels]
            summary = seg.groupby("Segment").mean(numeric_only=True).reset_index()
            summary.insert(1, "Records / customers", seg.groupby("Segment").size().values)
            assignments = pd.DataFrame({"Customer / row": tmp.index.astype(str), "Segment": [f"Segment {x+1}" for x in labels]})
            add("Customer Segmentation", f"Created {k} exploratory clusters from {tmp.shape[1]} numeric feature(s) and {tmp.shape[0]} customer/row records.", assignments.merge(summary, on="Segment", how="left"), "Clusters are exploratory and depend on selected numeric features. Validate business meaning and stability before operationalising. Missing feature values were median-imputed for clustering only; raw data remain unchanged unless the separate cleaning options were applied.")
        except Exception as exc:
            add("Customer Segmentation", f"Segmentation could not run: {exc}", note="Check numeric fields and install scikit-learn. Consider rule-based segments as an alternative.")
    elif col("customer_id") is not None and col("amount") is not None:
        temp = pd.DataFrame({"customer": df[mapping["customer_id"]], "amount": clean_for_numeric(df[mapping["amount"]])}).dropna()
        if len(temp):
            vals = temp.groupby("customer")["amount"].sum()
            try:
                vals = vals[vals >= 0]
                labels = pd.qcut(vals.rank(method="first"), q=min(3, len(vals)), labels=["Low value", "Medium value", "High value"][:min(3, len(vals))])
                add("Customer Segmentation", "Created exploratory customer-value tiers from observed revenue only.", pd.DataFrame({"Customer ID": vals.index.astype(str), "Observed revenue": vals.values, "Segment": labels.astype(str)}), "This is a one-dimensional value segmentation, not a multidimensional behavioural cluster model.")
            except Exception:
                pass

    # Feedback sentiment: VADER if installed, transparent lexicon fallback otherwise.
    feedback = col("feedback_text")
    if feedback is not None:
        texts = feedback.dropna().astype(str)
        texts = texts[texts.str.strip().ne("")]
        if len(texts):
            labels = []
            scores = []
            method = "simple keyword fallback"
            try:
                from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
                analyzer = SentimentIntensityAnalyzer()
                scores = [analyzer.polarity_scores(t)["compound"] for t in texts]
                method = "VADER lexicon-based sentiment"
            except Exception:
                pos_words = {"good", "great", "excellent", "happy", "helpful", "easy", "love", "perfect", "fast", "satisfied", "recommend", "thanks", "thank"}
                neg_words = {"bad", "poor", "terrible", "hate", "slow", "difficult", "angry", "broken", "late", "unsatisfied", "refund", "problem", "issue", "worst"}
                for t in texts:
                    words = set(re.findall(r"[a-z']+", t.casefold()))
                    p, n = len(words & pos_words), len(words & neg_words)
                    scores.append((p-n)/max(1, p+n))
            for score in scores:
                labels.append("Positive" if score >= 0.05 else "Negative" if score <= -0.05 else "Neutral / mixed")
            detail = pd.DataFrame({"Feedback": texts.values, "Sentiment score": scores, "Sentiment label": labels})
            add("Feedback Sentiment", f"Analysed {len(detail)} text responses using {method}. Results: {detail['Sentiment label'].value_counts().to_dict()}.", detail.head(250), "Lexicon sentiment is a rough screening aid, not an understanding of sarcasm, context, language variation or all languages. Human review is recommended for high-impact actions.")

    # Behaviour summary
    if col("customer_id") is not None and (col("txn_date") is not None or col("amount") is not None or col("event_type") is not None):
        tmp = pd.DataFrame({"customer": df[mapping["customer_id"]]})
        tmp["amount"] = clean_for_numeric(col("amount")) if col("amount") is not None else np.nan
        tmp["date"] = parse_dates(col("txn_date")) if col("txn_date") is not None else pd.NaT
        tmp["event"] = col("event_type").astype(str) if col("event_type") is not None else "record"
        tmp = tmp.dropna(subset=["customer"])
        agg = tmp.groupby("customer").agg(records=("customer", "size"), total_observed_amount=("amount", "sum"), average_record_amount=("amount", "mean"), last_activity=("date", "max"), distinct_event_types=("event", "nunique"))
        if agg["last_activity"].notna().any():
            cutoff = agg["last_activity"].max()
            agg["days_since_last_activity"] = (cutoff - agg["last_activity"]).dt.days
        add("Customer Behaviour", f"Built a customer-level summary for {len(agg)} customer IDs using the mapped fields.", agg.sort_values("records", ascending=False).head(250).reset_index().rename(columns={"customer": "Customer ID"}), "A row is treated as a transaction/activity record. Confirm the granularity of the source file; repeated snapshots can make record counts misleading.")

    # Loyalty
    if col("customer_id") is not None and col("txn_date") is not None:
        tmp = df[[mapping["customer_id"], mapping["txn_date"]]].copy()
        tmp.columns = ["customer", "date"]
        tmp["date"] = parse_dates(tmp["date"])
        tmp = tmp.dropna()
        tx_counts = tmp.groupby("customer").size()
        repeat_rate = float((tx_counts >= 2).mean() * 100) if len(tx_counts) else np.nan
        add("Customer Loyalty (repeat-purchase proxy)", f"{repeat_rate:.1f}% of customers have at least two records in this dataset; median records/customer: {tx_counts.median():.1f}." if len(tx_counts) else "No usable customer/date records.", tx_counts.value_counts().sort_index().rename_axis("Recorded transactions per customer").reset_index(name="Customers"), "Repeat purchase is a loyalty proxy, not a complete loyalty measure. If one customer can have multiple rows per order, map an order ID or deduplicate at the correct transaction grain.")
    elif col("customer_id") is not None and col("loyalty_flag") is not None:
        vals = col("loyalty_flag").dropna().astype(str).value_counts().rename_axis("Loyalty flag").reset_index(name="Rows")
        add("Customer Loyalty", f"Summarised {len(vals)} observed membership categories.", vals, "Membership status alone does not prove repeat purchase, retention or loyalty strength.")

    # CES
    s = col("ces_score")
    if s is not None:
        x = clean_for_numeric(s).dropna()
        if len(x):
            direction = options.get("ces_direction", "Higher score = more effort")
            add("Customer Effort Score (CES)", f"Mean CES score: {x.mean():.2f} (n={len(x)}); median {x.median():.2f}; direction configured as '{direction}'.", x.value_counts().sort_index().rename_axis("Score").reset_index(name="Responses"), "CES scales differ; verify scale direction and response coding before comparing teams or periods.")
        else:
            add("Customer Effort Score (CES)", "Mapped column contains no usable numeric effort scores.")

    # NPS
    s = col("nps_score")
    if s is not None:
        x = clean_for_numeric(s).dropna()
        valid = x[(x >= 0) & (x <= 10)]
        if len(valid):
            promoters = int((valid >= 9).sum())
            passives = int(((valid >= 7) & (valid <= 8)).sum())
            detractors = int((valid <= 6).sum())
            nps = (promoters - detractors) / len(valid) * 100
            dist = pd.DataFrame({"Group": ["Promoters (9–10)", "Passives (7–8)", "Detractors (0–6)"], "Count": [promoters, passives, detractors], "Percent": [promoters/len(valid)*100, passives/len(valid)*100, detractors/len(valid)*100]})
            add("Net Promoter Score (NPS)", f"NPS: {nps:.1f} (valid 0–10 responses n={len(valid)}). Promoters {promoters/len(valid)*100:.1f}%, passives {passives/len(valid)*100:.1f}%, detractors {detractors/len(valid)*100:.1f}%.", dist, "NPS is percentage of promoters minus percentage of detractors, not the average of the 0–10 responses. Scores outside 0–10 were excluded and should be investigated.")
        else:
            add("Net Promoter Score (NPS)", "No usable score in the 0–10 range.")

    # CAC
    manual_spend = options.get("manual_spend")
    manual_new = options.get("manual_new_customers")
    cac = np.nan
    cac_note = ""
    if manual_spend is not None and manual_new is not None and manual_new > 0:
        cac = float(manual_spend) / float(manual_new)
        cac_note = "Calculated from user-entered aggregate acquisition spend and new-customer count. Verify both refer to the same period and channel scope."
    elif col("acquisition_spend") is not None and col("new_customers_count") is not None:
        if not options.get("cac_inputs_confirmed", False):
            add("Customer Acquisition Cost (CAC)", "CAC not calculated: source columns are present, but additive granularity and period alignment have not been confirmed.", note="Confirm that spend values are not repeated across transaction/customer rows and that spend and newly acquired-customer counts share the same period, channels, currency and attribution scope. Alternatively enter verified aggregate totals in the analysis options.")
        else:
            spend = clean_for_numeric(col("acquisition_spend")).sum(min_count=1)
            new_count = clean_for_numeric(col("new_customers_count")).sum(min_count=1)
            if pd.notna(spend) and pd.notna(new_count) and new_count > 0:
                cac = float(spend) / float(new_count)
                cac_note = "Calculated from user-confirmed additive row values. Recheck period, channel, currency and attribution scope before business use."
    elif col("acquisition_spend") is not None and col("customer_id") is not None and col("new_customer_flag") is not None:
        if not options.get("cac_inputs_confirmed", False):
            add("Customer Acquisition Cost (CAC)", "CAC not calculated: source spend and new-customer flag are mapped, but row granularity and spend attribution have not been confirmed.", note="Confirm that spend is not repeated per transaction/customer row and that the new-customer flag identifies unique acquisitions over the same spend period.")
        else:
            spends = clean_for_numeric(col("acquisition_spend")).sum(min_count=1)
            is_new = truthy_flag(col("new_customer_flag"), options.get("positive_flag_values", "yes,true,1,new"))
            new_n = df.loc[is_new, mapping["customer_id"]].nunique()
            if pd.notna(spends) and new_n > 0:
                cac = float(spends) / int(new_n)
                cac_note = "Calculated from user-confirmed spend aggregation and unique new-customer identities. Verify same period, channel and attribution scope."
    if pd.notna(cac):
        add("Customer Acquisition Cost (CAC)", f"Estimated CAC: {cac:,.2f} spend units per new customer.", note=cac_note)
        meta["cac_estimate"] = float(cac)
    elif col("acquisition_spend") is not None or manual_spend is not None:
        add("Customer Acquisition Cost (CAC)", "Acquisition spend is available, but a comparable new-customer count was not available or was zero. CAC was not calculated.", note="Supply new customers acquired in the same date range and matching scope. Do not divide by all contacts or all leads.")

    # CLV:CAC ratio, with observed historical revenue proxy only
    if col("customer_id") is not None and col("amount") is not None and pd.notna(cac) and cac > 0:
        temp = pd.DataFrame({"customer": df[mapping["customer_id"]], "amount": clean_for_numeric(df[mapping["amount"]])}).dropna()
        clv = temp.groupby("customer")["amount"].sum().mean() if len(temp) else np.nan
        if pd.notna(clv):
            add("CLV:CAC and CAC:CLV ratios", f"Historical observed revenue/customer ÷ CAC = {clv/cac:.2f}:1. Reverse ratio CAC ÷ historical observed revenue/customer = {cac/clv:.4f} when observed value is non-zero.", note="This is a historical revenue proxy, not predictive CLV or contribution-margin-based unit economics. Compare consistent cohorts and periods; include margin, retention horizon and cost definitions for a stronger model.")

    # FCR
    s = col("fcr_flag")
    if s is not None:
        vals = s.dropna()
        if len(vals):
            yes = truthy_flag(vals, options.get("positive_fcr_labels", "yes,true,1,yes fcr,first contact,first-contact"))
            add("First Contact Resolution (FCR)", f"Observed FCR rate: {yes.mean()*100:.1f}% ({int(yes.sum())}/{len(vals)} non-missing flags).", vals.astype(str).value_counts().rename_axis("Flag value").reset_index(name="Rows"), "Confirm which values indicate first-contact resolution. An ordinary 'resolved' status is not necessarily FCR.")

    # Ticket time: calculate resolution/first response if both timestamps.
    created = col("ticket_created")
    resolved = col("ticket_resolved")
    response = col("first_response")
    if created is not None and (resolved is not None or response is not None):
        ct = parse_dates(created)
        detail = pd.DataFrame({"ticket_created": ct})
        if resolved is not None:
            rt = parse_dates(resolved)
            hours = (rt - ct).dt.total_seconds() / 3600
            valid = hours[(hours >= 0) & np.isfinite(hours)]
            detail["resolution_hours"] = hours
            if len(valid):
                add("Average Ticket / Response Time", f"Average resolution time: {valid.mean():.2f} hours (median {valid.median():.2f}; n={len(valid)}).", detail.loc[valid.index].head(250), "Negative durations and missing timestamps were excluded from the average and should be audited. This metric measures elapsed time, not agent handling time.")
        elif response is not None:
            # Detect whether the mapped column is a duration or timestamp.
            r_numeric = clean_for_numeric(response)
            if r_numeric.notna().sum() > 0 and r_numeric.notna().mean() > 0.7:
                x = r_numeric[r_numeric >= 0]
                add("Average First Response Time", f"Mean first-response duration: {x.mean():.2f} in the source column's units (n={len(x)}).", x.describe().to_frame("value").reset_index().rename(columns={"index": "statistic"}), "Source units are not known automatically; confirm if this is minutes, hours or another unit.")
            else:
                rt = parse_dates(response)
                hours = (rt - ct).dt.total_seconds() / 3600
                valid = hours[(hours >= 0) & np.isfinite(hours)]
                if len(valid):
                    add("Average First Response Time", f"Average elapsed time to first response: {valid.mean():.2f} hours (median {valid.median():.2f}; n={len(valid)}).", valid.describe().to_frame("hours").reset_index().rename(columns={"index": "statistic"}), "Negative durations and missing timestamps were excluded. Confirm the mapped column contains first-response timestamps.")

    # Price sensitivity (descriptive association only)
    price, qty = col("unit_price"), col("quantity")
    if price is not None and qty is not None:
        tmp = pd.DataFrame({"price": clean_for_numeric(price), "quantity": clean_for_numeric(qty)})
        tmp = tmp.replace([np.inf, -np.inf], np.nan).dropna()
        tmp = tmp[(tmp["price"] > 0) & (tmp["quantity"] >= 0)]
        if len(tmp) >= 3:
            corr = tmp["price"].corr(tmp["quantity"]) if tmp["price"].nunique() > 1 and tmp["quantity"].nunique() > 1 else np.nan
            add("Price Sensitivity (descriptive)", f"Observed price/quantity Pearson correlation: {corr:.3f} across {len(tmp)} records." if pd.notna(corr) else f"{len(tmp)} valid records but insufficient variation to calculate correlation.", tmp.groupby("price").agg(records=("quantity", "size"), total_quantity=("quantity", "sum"), mean_quantity=("quantity", "mean")).reset_index().sort_values("price"), "A simple correlation is not causal price elasticity. Product mix, promotion, seasonality, stock availability and customer selection can confound the relationship. Price-sensitivity estimation needs meaningful price variation and controls or a designed experiment.")

    # RFM
    if col("customer_id") is not None and col("txn_date") is not None and col("amount") is not None:
        temp = pd.DataFrame({"customer": df[mapping["customer_id"]], "date": parse_dates(df[mapping["txn_date"]]), "amount": clean_for_numeric(df[mapping["amount"]])})
        temp = temp.dropna(subset=["customer", "date", "amount"])
        if len(temp):
            snap = temp["date"].max().normalize()
            rec = (snap - temp.groupby("customer")["date"].max().dt.normalize()).dt.days
            if col("order_id") is not None:
                freq = temp.join(df[[mapping["order_id"]]].rename(columns={mapping["order_id"]: "order_id"})).groupby("customer")["order_id"].nunique()
            else:
                freq = temp.groupby("customer").size()
            mon = temp.groupby("customer")["amount"].sum()
            rfm = pd.concat([rec.rename("Recency days"), freq.rename("Frequency orders/records"), mon.rename("Monetary observed amount")], axis=1).dropna()
            if len(rfm) >= 3:
                # Rank-based quintiles avoid qcut failures from many ties.
                n_bins = min(5, len(rfm))
                r_score = pd.qcut(rfm["Recency days"].rank(method="first", ascending=False), q=n_bins, labels=False, duplicates="drop") + 1
                f_score = pd.qcut(rfm["Frequency orders/records"].rank(method="first"), q=n_bins, labels=False, duplicates="drop") + 1
                m_score = pd.qcut(rfm["Monetary observed amount"].rank(method="first"), q=n_bins, labels=False, duplicates="drop") + 1
                rfm["R score"] = r_score.astype(int)
                rfm["F score"] = f_score.astype(int)
                rfm["M score"] = m_score.astype(int)
                rfm["RFM code"] = rfm[["R score", "F score", "M score"]].astype(str).agg("".join, axis=1)
                add("RFM Analysis", f"RFM scored {len(rfm)} customers using snapshot date {snap.date()}; frequency uses distinct order IDs where mapped, otherwise row count as transaction proxy.", rfm.sort_values(["R score", "F score", "M score"], ascending=False).head(500).reset_index().rename(columns={"customer": "Customer ID"}), "Scores are relative to this dataset and are not directly comparable across periods without a consistent scoring strategy. Monetary sums retain negative values such as refunds.")
            else:
                add("RFM Analysis", "Fewer than three customers with complete IDs, dates and amounts; RFM scoring was not run.", note="Provide more customer records or inspect missing values. Recency, frequency and monetary summaries can still be generated separately.")

    # General descriptive analytics always available
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if numeric_cols:
        desc = df[numeric_cols].describe().T.reset_index(names="column")
        add("Additional: Descriptive Statistics", f"Generated descriptive statistics for {len(numeric_cols)} numeric columns.", desc, "Inspect units, missingness and unusual values before interpreting distributions.")

    if col("channel") is not None:
        ch = col("channel")
        if col("amount") is not None:
            tmp = pd.DataFrame({"channel": ch, "amount": clean_for_numeric(col("amount"))}).dropna()
            if len(tmp):
                tab = tmp.groupby("channel").agg(records=("amount", "size"), total_amount=("amount", "sum"), average_amount=("amount", "mean")).reset_index().sort_values("total_amount", ascending=False)
                add("Additional: Channel Performance", f"Compared {tab['channel'].nunique()} channels on recorded amount.", tab, "Recorded sales/revenue by channel is not marketing ROI unless acquisition spend and attribution are also available.")
        else:
            add("Additional: Channel Mix", f"Summarised {ch.nunique(dropna=True)} observed channel categories.", ch.value_counts(dropna=False).rename_axis("Channel").reset_index(name="Rows"), "Counts describe the uploaded data only; they are not conversion rates unless denominators are available.")

    return results, meta


def make_report_text(filename: str, original_df: pd.DataFrame, cleaned_df: pd.DataFrame, missing: pd.DataFrame,
                     outliers: pd.DataFrame, audit: list[str], feas: pd.DataFrame,
                     results: dict[str, dict[str, Any]]) -> str:
    lines = [
        "CUSTOMER ANALYTICS INTELLIGENCE REPORT",
        "=" * 44,
        f"Source file: {filename}",
        f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
        f"Original shape: {original_df.shape[0]} rows × {original_df.shape[1]} columns",
        f"Analysis shape after selected cleaning: {cleaned_df.shape[0]} rows × {cleaned_df.shape[1]} columns",
        "",
        "DATA QUALITY AUDIT",
        "-" * 20,
        f"Exact duplicate rows in original file: {int(original_df.duplicated().sum())}",
        f"Columns with at least one missing value: {int((missing['missing_count'] > 0).sum()) if len(missing) else 0}",
        "Cleaning actions:",
    ]
    lines.extend([f"- {x}" for x in audit])
    if len(outliers):
        lines.extend(["", "OUTLIER SCREEN (IQR RULE; FLAGS ARE NOT PROOF OF ERROR)", "-" * 45])
        for _, row in outliers[outliers.get("flagged_outliers", pd.Series(dtype=int)) > 0].iterrows():
            lines.append(f"- {row['column']}: {int(row['flagged_outliers'])} flagged value(s), bounds [{row['iqr_low']}, {row['iqr_high']}]")
    lines.extend(["", "FEASIBILITY ASSESSMENT", "-" * 24])
    for _, row in feas.iterrows():
        lines.append(f"- {row['Analysis']}: {row['Status']}. {row['Reason / requirement']} Alternative: {row['Alternative']}")
    lines.extend(["", "RESULTS", "-" * 7])
    for name, result in results.items():
        lines.append(f"\n{name}\n{result['summary']}")
        if result.get("note"):
            lines.append(f"Limitations: {result['note']}")
    lines.extend(["", "IMPORTANT INTERPRETATION RULES", "-" * 29,
                  "1. Historical revenue/customer is not the same as predictive CLV.",
                  "2. Inactivity is a churn-risk proxy unless an explicit churn outcome is available.",
                  "3. Correlation does not establish causality; price sensitivity requires suitable variation and controls or an experiment.",
                  "4. Outliers are flagged, not automatically treated as errors. Review the audit and compare results before/after any alteration.",
                  "5. Analyses are conditional on the correctness, representativeness and granularity of the uploaded data."])
    return "\n".join(lines)


