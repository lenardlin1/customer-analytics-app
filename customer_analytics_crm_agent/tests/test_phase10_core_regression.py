import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
import pytest
from customer_analytics.profiling import profile_dataframe
from customer_analytics.schema import suggest_column_mapping
from customer_analytics.feasibility import assess_feasibility
from customer_analytics.analytics import calculate_rfm, calculate_aov, calculate_repeat_purchase, AnalysisError


def example_transactions():
    return pd.DataFrame({
        "Customer_ID": ["C1", "C1", "C2", "C3"],
        "Order_ID": ["O1", "O2", "O3", "O4"],
        "Order_Date": ["2025-01-01", "2025-02-01", "2025-01-15", "2025-02-15"],
        "Order_Value": [100.0, 50.0, 80.0, 20.0],
        "Product_ID": ["P1", "P2", "P1", "P3"],
    })


def test_profile_counts_duplicates_and_missing():
    df = pd.DataFrame({"x": [1, 1, None], "y": ["a", "a", "b"]})
    profile = profile_dataframe(df)
    assert profile["row_count"] == 3
    assert profile["column_count"] == 2
    assert profile["exact_duplicate_rows"] == 1
    assert profile["columns"][0]["missing_count"] == 1


def test_mapping_detects_common_transaction_fields():
    mapping = suggest_column_mapping(example_transactions())
    assert mapping["customer_id"] == "Customer_ID"
    assert mapping["order_id"] == "Order_ID"
    assert mapping["order_date"] == "Order_Date"
    assert mapping["revenue"] == "Order_Value"


def test_registry_has_35_analysis_definitions_and_nps_not_feasible_without_score():
    df = example_transactions()
    mapping = suggest_column_mapping(df)
    results = assess_feasibility(df, mapping)
    assert len(results) == 35
    nps = next(item for item in results if item["analysis_id"] == "CA-011")
    assert nps["feasibility_status"] == "Not feasible"


def test_rfm_metrics_are_deterministic():
    df = example_transactions()
    mapping = suggest_column_mapping(df)
    result = calculate_rfm(df, mapping, reference_date="2025-03-01")
    assert result.summary["customers_analyzed"] == 3
    c1 = result.table[result.table["Customer_ID"] == "C1"].iloc[0]
    assert c1["Frequency"] == 2
    assert c1["Monetary_Value"] == 150.0
    assert c1["Recency_Days"] == 28.0


def test_aov_computes_order_level_values():
    df = example_transactions()
    result = calculate_aov(df, suggest_column_mapping(df))
    assert result.summary["orders_analyzed"] == 4
    assert result.summary["total_order_value"] == 250.0
    assert result.summary["average_order_value"] == 62.5


def test_aov_refuses_ambiguous_multirow_order_values():
    df = pd.DataFrame({"Order_ID": ["O1", "O1"], "Order_Value": [10.0, 15.0]})
    with pytest.raises(AnalysisError, match="Confirm whether revenue is per line item"):
        calculate_aov(df, suggest_column_mapping(df))


def test_repeat_purchase_rate():
    df = example_transactions()
    result = calculate_repeat_purchase(df, suggest_column_mapping(df))
    assert result.summary["eligible_customers"] == 3
    assert result.summary["repeat_customers"] == 1
    assert result.summary["repeat_purchase_rate_pct"] == pytest.approx(33.33, abs=0.01)


def test_duplicate_header_profile_does_not_crash():
    df = pd.DataFrame([[1, 2], [3, 4]], columns=["value", "value"])
    profile = profile_dataframe(df)
    assert profile["column_count"] == 2
    assert "Duplicate column names were detected; resolve before analysis." in profile["warnings"]


def test_rfm_one_customer_returns_neutral_scores_not_crash():
    df = pd.DataFrame({
        "Customer_ID": ["C1"],
        "Order_ID": ["O1"],
        "Order_Date": ["2025-01-01"],
        "Order_Value": [100.0],
    })
    result = calculate_rfm(df, suggest_column_mapping(df), reference_date="2025-02-01")
    row = result.table.iloc[0]
    assert row["R_Score"] == 3
    assert row["F_Score"] == 3
    assert row["M_Score"] == 3
    assert any("Only one customer" in warning for warning in result.warnings)


def test_profile_adds_date_coverage_for_date_named_columns():
    df = pd.DataFrame({"Order_Date": ["2025-01-01", "bad-date", None]})
    profile = profile_dataframe(df)
    record = profile["columns"][0]
    assert record["inferred_kind"] == "datetime candidate"
    assert record["date_parseable_count"] == 1
    assert record["date_parse_failure_count"] == 1


def test_aov_refuses_repeated_identical_order_amounts_until_granularity_is_confirmed():
    df = pd.DataFrame({"Order_ID": ["O1", "O1"], "Order_Value": [10.0, 10.0]})
    with pytest.raises(AnalysisError, match="Repeated order IDs were detected"):
        calculate_aov(df, suggest_column_mapping(df))
