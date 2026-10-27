"""Deterministic MVP analyses. All monetary and frequency assumptions are exposed."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import math
import pandas as pd


class AnalysisError(ValueError):
    """Raised if an analysis cannot be completed without an unsafe assumption."""

@dataclass
class AnalysisResult:
    analysis_id: str
    name: str
    summary: dict[str, Any]
    table: pd.DataFrame | None = None
    warnings: list[str] | None = None


def _actual(mapping: dict[str, str | None], semantic: str) -> str:
    col = mapping.get(semantic)
    if not col:
        raise AnalysisError(f"Required field '{semantic}' is not mapped.")
    return col


def _parse_dates(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=True)


def _customer_order_revenue(df: pd.DataFrame, mapping: dict[str, str | None]) -> tuple[pd.DataFrame, list[str]]:
    customer = _actual(mapping, "customer_id")
    revenue = _actual(mapping, "revenue")
    order = mapping.get("order_id")
    date = mapping.get("order_date")
    needed = [customer, revenue] + ([order] if order else []) + ([date] if date else [])
    frame = df[needed].copy()
    original_rows = len(frame)
    original_revenue = frame[revenue].copy()
    frame[revenue] = pd.to_numeric(frame[revenue], errors="coerce")
    missing_customer_rows = int(frame[customer].isna().sum())
    missing_revenue_rows = int(original_revenue.isna().sum())
    invalid_revenue_rows = int((original_revenue.notna() & frame[revenue].isna()).sum())
    frame = frame.dropna(subset=[customer, revenue])
    warnings: list[str] = []
    if missing_customer_rows:
        warnings.append(f"{missing_customer_rows} rows with missing customer IDs were excluded from customer-level revenue analysis.")
    if missing_revenue_rows:
        warnings.append(f"{missing_revenue_rows} rows with missing revenue values were excluded from revenue aggregation.")
    if invalid_revenue_rows:
        warnings.append(f"{invalid_revenue_rows} rows with non-numeric revenue values were excluded; verify the currency/number format.")
    if original_rows and len(frame) == 0:
        raise AnalysisError("No valid customer IDs and numeric revenue values remain after data validation.")
    if not order:
        warnings.append("No order ID is mapped; revenue is summed by row. Verify that repeated order totals are not duplicated across order-item rows.")
    if order:
        missing_order_ids = int(frame[order].isna().sum())
        if missing_order_ids:
            frame = frame.dropna(subset=[order])
            warnings.append(f"{missing_order_ids} rows with missing order IDs were excluded from order-based revenue aggregation.")
        if frame.duplicated(subset=[order]).any():
            raise AnalysisError(
                "Repeated order IDs were detected. They may represent order-item rows or repeated order totals. "
                "Confirm the order granularity and aggregate the file accordingly before computing customer value. "
                "This MVP will not guess whether to sum or deduplicate these rows."
            )
    return frame, warnings


def _quintile_score(values: pd.Series, *, higher_is_better: bool) -> pd.Series:
    """Assign tied values the same score; avoid implying differentiation when none exists."""
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().sum() <= 1 or numeric.nunique(dropna=True) <= 1:
        return pd.Series(3, index=values.index, dtype="int64")
    percentile_rank = numeric.rank(method="average", pct=True, ascending=higher_is_better)
    score = (percentile_rank * 5).apply(lambda value: int(math.ceil(value)) if pd.notna(value) else 3)
    return score.clip(lower=1, upper=5).astype("int64")


def calculate_rfm(df: pd.DataFrame, mapping: dict[str, str | None], reference_date: str | None = None) -> AnalysisResult:
    customer = _actual(mapping, "customer_id")
    order_date = _actual(mapping, "order_date")
    revenue = _actual(mapping, "revenue")
    order = mapping.get("order_id")
    cols = [customer, order_date, revenue] + ([order] if order else [])
    frame = df[cols].copy()
    frame[order_date] = _parse_dates(frame[order_date])
    frame[revenue] = pd.to_numeric(frame[revenue], errors="coerce")
    original_rows = len(frame)
    frame = frame.dropna(subset=[customer, order_date, revenue])
    warnings: list[str] = []
    if order:
        missing_order_ids = int(frame[order].isna().sum())
        if missing_order_ids:
            warnings.append(f"{missing_order_ids} rows with missing order IDs were excluded to keep RFM transaction counts consistent.")
            frame = frame.dropna(subset=[order])
        if frame[order].duplicated().any():
            raise AnalysisError(
                "Repeated order IDs were detected. Confirm whether rows are order items or repeated order totals, "
                "then aggregate to one validated order record before RFM. This MVP will not guess how to combine them."
            )
    if frame.empty:
        raise AnalysisError("No valid rows remain with customer ID, order date, revenue and a usable transaction identifier.")
    if reference_date:
        ref = pd.Timestamp(reference_date)
        if ref.tzinfo is None:
            ref = ref.tz_localize("UTC")
        else:
            ref = ref.tz_convert("UTC")
    else:
        ref = frame[order_date].max().normalize() + pd.Timedelta(days=1)
    if ref < frame[order_date].max():
        raise AnalysisError("Reference date is earlier than the latest transaction date.")
    grouped = frame.groupby(customer, dropna=False)
    frequency = grouped[order].nunique() if order else grouped.size()
    result = pd.DataFrame({
        "Recency_Days": (ref - grouped[order_date].max()).dt.total_seconds() / 86400,
        "Frequency": frequency,
        "Monetary_Value": grouped[revenue].sum(),
    }).reset_index(names="Customer_ID")
    result["Recency_Days"] = result["Recency_Days"].round(2)
    result["R_Score"] = _quintile_score(result["Recency_Days"], higher_is_better=False)
    result["F_Score"] = _quintile_score(result["Frequency"], higher_is_better=True)
    result["M_Score"] = _quintile_score(result["Monetary_Value"], higher_is_better=True)
    result["RFM_Score"] = result[["R_Score", "F_Score", "M_Score"]].astype(int).astype(str).agg("".join, axis=1)
    if not order:
        warnings.append("Frequency is row count because no order ID was mapped; valid only if each row is one purchase transaction.")
    if result["Customer_ID"].nunique() <= 1:
        warnings.append("Only one customer is represented; RFM scores are neutral placeholders and cannot distinguish customers.")
    if (frame[revenue] < 0).any():
        warnings.append("Negative monetary transactions are present; confirm whether these are refunds/credits and whether net monetary value is the intended definition.")
    if len(frame) < original_rows:
        warnings.append(f"{original_rows - len(frame)} rows were excluded because an essential customer, date, revenue or transaction identifier was missing/invalid.")
    return AnalysisResult("CA-017", "RFM analysis", {
        "customers_analyzed": int(result["Customer_ID"].nunique()),
        "reference_date_utc": str(ref),
        "frequency_definition": "Distinct order IDs" if order else "Row count proxy; verify one row per purchase",
        "monetary_definition": f"Sum of {revenue}",
    }, result.sort_values(["R_Score", "F_Score", "M_Score"], ascending=False), warnings)


def calculate_customer_revenue(df: pd.DataFrame, mapping: dict[str, str | None]) -> AnalysisResult:
    frame, warnings = _customer_order_revenue(df, mapping)
    customer = _actual(mapping, "customer_id")
    revenue = _actual(mapping, "revenue")
    table = frame.groupby(customer, dropna=False)[revenue].sum().sort_values(ascending=False).reset_index()
    table.columns = ["Customer_ID", "Historical_Revenue"]
    return AnalysisResult("CA-018", "Historical revenue by customer", {
        "customers_analyzed": int(table["Customer_ID"].nunique()),
        "total_historical_revenue": float(table["Historical_Revenue"].sum()),
        "value_definition": "Historical revenue; not profit or predictive CLV",
    }, table, warnings)


def calculate_aov(df: pd.DataFrame, mapping: dict[str, str | None]) -> AnalysisResult:
    order = _actual(mapping, "order_id")
    revenue = _actual(mapping, "revenue")
    frame = df[[order, revenue]].copy()
    original_revenue = frame[revenue].copy()
    frame[revenue] = pd.to_numeric(frame[revenue], errors="coerce")
    missing_order_rows = int(frame[order].isna().sum())
    missing_revenue_rows = int(original_revenue.isna().sum())
    invalid_revenue_rows = int((original_revenue.notna() & frame[revenue].isna()).sum())
    frame = frame.dropna(subset=[order, revenue])
    if frame.empty:
        raise AnalysisError("No valid order IDs and numeric revenue values were found.")
    warnings: list[str] = []
    if missing_order_rows:
        warnings.append(f"{missing_order_rows} rows with missing order IDs were excluded from AOV.")
    if missing_revenue_rows:
        warnings.append(f"{missing_revenue_rows} rows with missing revenue values were excluded from AOV.")
    if invalid_revenue_rows:
        warnings.append(f"{invalid_revenue_rows} rows with non-numeric revenue values were excluded from AOV; verify the currency/number format.")
    if frame.duplicated(subset=[order]).any():
        raise AnalysisError(
            "Repeated order IDs were detected. Confirm whether revenue is per line item or total order value, "
            "then aggregate to one validated order record before calculating AOV. This MVP will not guess."
        )
    order_values = frame.groupby(order)[revenue].sum()
    return AnalysisResult("CA-019", "Average Order Value", {
        "orders_analyzed": int(order_values.size),
        "total_order_value": float(order_values.sum()),
        "average_order_value": float(order_values.mean()),
        "median_order_value": float(order_values.median()),
        "value_definition": "One validated monetary value per order",
    }, order_values.rename("Order_Value").reset_index(), warnings)


def calculate_repeat_purchase(df: pd.DataFrame, mapping: dict[str, str | None]) -> AnalysisResult:
    customer = _actual(mapping, "customer_id")
    order_date = _actual(mapping, "order_date")
    order = mapping.get("order_id")
    cols = [customer, order_date] + ([order] if order else [])
    frame = df[cols].copy()
    original_rows = len(frame)
    original_dates = frame[order_date].copy()
    frame[order_date] = _parse_dates(frame[order_date])
    missing_customer_rows = int(frame[customer].isna().sum())
    missing_date_rows = int(original_dates.isna().sum())
    invalid_date_rows = int((original_dates.notna() & frame[order_date].isna()).sum())
    frame = frame.dropna(subset=[customer, order_date])
    excluded_before_orders = original_rows - len(frame)
    if frame.empty:
        raise AnalysisError("No valid customer IDs and transaction dates were found.")
    if order:
        missing_order_ids = int(frame[order].isna().sum())
        frame = frame.dropna(subset=[order])
        if frame.empty:
            raise AnalysisError("No valid order IDs remain for repeat-purchase analysis.")
        counts = frame.groupby(customer)[order].nunique()
        frequency_definition = "distinct order IDs"
    else:
        counts = frame.groupby(customer).size()
        frequency_definition = "row count proxy; verify one row per order"
    eligible = int(counts.size)
    repeat = int((counts >= 2).sum())
    warnings = ["Repeat purchase is defined as at least two recorded orders during the observed data window."]
    if missing_customer_rows:
        warnings.append(f"{missing_customer_rows} rows with missing customer IDs were excluded.")
    if missing_date_rows:
        warnings.append(f"{missing_date_rows} rows with missing transaction dates were excluded.")
    if invalid_date_rows:
        warnings.append(f"{invalid_date_rows} rows with invalid transaction dates were excluded; verify the date format.")
    if excluded_before_orders and not (missing_customer_rows or missing_date_rows or invalid_date_rows):
        warnings.append(f"{excluded_before_orders} rows were excluded because customer identity or transaction date was unavailable.")
    if order and missing_order_ids:
        warnings.append(f"{missing_order_ids} rows with missing order IDs were excluded from frequency counts.")
    return AnalysisResult("CA-020", "Repeat-purchase analysis", {
        "eligible_customers": eligible,
        "repeat_customers": repeat,
        "repeat_purchase_rate_pct": round(100 * repeat / eligible, 2) if eligible else None,
        "frequency_definition": frequency_definition,
        "window": f"{frame[order_date].min()} to {frame[order_date].max()}",
    }, counts.rename("Purchase_Count").reset_index(), warnings)


def calculate_revenue_concentration(df: pd.DataFrame, mapping: dict[str, str | None]) -> AnalysisResult:
    base, warnings = _customer_order_revenue(df, mapping)
    customer = _actual(mapping, "customer_id")
    revenue = _actual(mapping, "revenue")
    values = base.groupby(customer, dropna=False)[revenue].sum().sort_values(ascending=False)
    total = float(values.sum())
    if total <= 0:
        raise AnalysisError("Total customer revenue is zero or negative; standard revenue-share concentration is not meaningful.")
    if (values < 0).any():
        raise AnalysisError("Some customers have negative net revenue. Review refunds/credits and define the concentration measure before calculating standard revenue shares.")
    top_n = max(1, int(len(values) * 0.10 + 0.999999))
    top_share = float(values.head(top_n).sum() / total)
    shares = values / total
    table = pd.DataFrame({"Customer_ID": values.index, "Historical_Revenue": values.values, "Revenue_Share_Pct": (shares.values * 100)})
    return AnalysisResult("CA-030", "Revenue concentration", {
        "customers_analyzed": int(values.size),
        "top_10pct_customer_count": top_n,
        "top_10pct_customer_revenue_share_pct": round(100 * top_share, 2),
        "hhi_on_customer_revenue_shares": float((shares ** 2).sum()),
        "total_historical_revenue": total,
    }, table, warnings + ["Concentration is a dependency indicator, not a churn prediction."])


SUPPORTED_ANALYSES = {
    "CA-017": calculate_rfm,
    "CA-018": calculate_customer_revenue,
    "CA-019": calculate_aov,
    "CA-020": calculate_repeat_purchase,
    "CA-030": calculate_revenue_concentration,
}


def run_analysis(analysis_id: str, df: pd.DataFrame, mapping: dict[str, str | None]) -> AnalysisResult:
    if analysis_id not in SUPPORTED_ANALYSES:
        raise AnalysisError(f"Analysis {analysis_id} is registered but not implemented in this MVP.")
    return SUPPORTED_ANALYSES[analysis_id](df, mapping)
