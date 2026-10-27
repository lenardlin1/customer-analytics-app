"""Phase 10 acceptance tests: end-to-end correctness, safety, and edge cases."""
from __future__ import annotations

from io import BytesIO
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import pytest

from customer_analytics.agent import assess_dataset, execute_approved_analyses
from customer_analytics.analytics import (
    AnalysisError,
    AnalysisResult,
    calculate_aov,
    calculate_customer_revenue,
    calculate_repeat_purchase,
    calculate_revenue_concentration,
    calculate_rfm,
)
from customer_analytics.feasibility import assess_feasibility, VALID_STATUSES
from customer_analytics.ingestion import load_uploaded_file, IngestionError
from customer_analytics.interpretation import deterministic_interpretation
from customer_analytics.profiling import profile_dataframe
from customer_analytics.reporting import build_markdown_report, dataframe_to_csv_bytes
from customer_analytics.result_validation import validate_result
from customer_analytics.schema import suggest_column_mapping


def transactions() -> pd.DataFrame:
    return pd.DataFrame({
        "Customer_ID": ["C1", "C1", "C2", "C3"],
        "Order_ID": ["O1", "O2", "O3", "O4"],
        "Order_Date": ["2025-01-01", "2025-02-01", "2025-01-15", "2025-02-15"],
        "Order_Value": [100.0, 50.0, 80.0, 20.0],
        "Product_ID": ["P1", "P2", "P1", "P3"],
    })


def test_end_to_end_assessment_execute_and_report_are_traceable():
    df = transactions()
    original = df.copy(deep=True)
    mapping = suggest_column_mapping(df)
    profile = profile_dataframe(df)
    definitions = {"rfm_definition", "period_definition", "repeat_definition", "order_granularity"}
    feasibility = assess_feasibility(df, mapping, confirmed_definitions=definitions)
    rfm_row = next(r for r in feasibility if r["analysis_id"] == "CA-017")
    assert rfm_row["feasibility_status"] in VALID_STATUSES

    # Explicit approval is required; only allowlisted implemented analyses should run.
    results, errors = execute_approved_analyses(
        df, mapping, feasibility, ["CA-017", "CA-999"]
    )
    assert len(results) == 1
    assert results[0].analysis_id == "CA-017"
    assert any(e["analysis_id"] == "CA-999" for e in errors)
    assert validate_result(results[0]) == []

    report = build_markdown_report("transactions.csv", "transactions", profile, feasibility, results)
    assert "Customer Analytics Report" in report
    assert "CA-017" in report
    assert "RFM analysis" in report
    assert "not profit or predictive CLV" not in report  # this is an RFM result, not a revenue result
    pd.testing.assert_frame_equal(df, original)


def test_dispatcher_blocks_non_feasible_analysis():
    df = transactions()
    mapping = suggest_column_mapping(df)
    feasibility = assess_feasibility(df, mapping)
    for item in feasibility:
        if item["analysis_id"] == "CA-017":
            item["feasibility_status"] = "Not feasible"
    results, errors = execute_approved_analyses(df, mapping, feasibility, ["CA-017"])
    assert results == []
    assert "blocked" in errors[0]["error"].lower()


def test_dispatcher_refuses_registered_but_unimplemented_analysis():
    df = transactions()
    mapping = suggest_column_mapping(df)
    feasibility = assess_feasibility(df, mapping)
    item = next(r for r in feasibility if r["analysis_id"] == "CA-011")
    item["feasibility_status"] = "Feasible"
    results, errors = execute_approved_analyses(df, mapping, feasibility, ["CA-011"])
    assert not results
    assert "not implemented" in errors[0]["error"].lower()


def test_customer_revenue_does_not_modify_source_and_matches_manual_total():
    df = transactions()
    original = df.copy(deep=True)
    result = calculate_customer_revenue(df, suggest_column_mapping(df))
    assert result.summary["total_historical_revenue"] == pytest.approx(250.0)
    assert result.summary["customers_analyzed"] == 3
    pd.testing.assert_frame_equal(df, original)


def test_repeat_purchase_explains_excluded_invalid_date_rows():
    df = pd.DataFrame({
        "Customer_ID": ["C1", "C1", "C2"],
        "Order_ID": ["O1", "O2", "O3"],
        "Order_Date": ["2025-01-01", "not-a-date", "2025-02-01"],
    })
    result = calculate_repeat_purchase(df, suggest_column_mapping(df))
    assert result.summary["eligible_customers"] == 2
    assert any("excluded" in warning.lower() for warning in result.warnings)


def test_invalid_revenue_rows_are_reported_not_silently_discarded():
    df = pd.DataFrame({
        "Customer_ID": ["C1", "C2", "C3"],
        "Order_ID": ["O1", "O2", "O3"],
        "Order_Value": [100, "not-money", None],
    })
    result = calculate_customer_revenue(df, suggest_column_mapping(df))
    assert result.summary["total_historical_revenue"] == pytest.approx(100.0)
    assert any("excluded" in warning.lower() for warning in result.warnings)


def test_aov_reports_invalid_revenue_rows():
    df = pd.DataFrame({"Order_ID": ["O1", "O2"], "Order_Value": [10, "bad"]})
    result = calculate_aov(df, suggest_column_mapping(df))
    assert result.summary["orders_analyzed"] == 1
    assert any("revenue" in warning.lower() and "excluded" in warning.lower() for warning in result.warnings)


def test_rfm_refuses_duplicate_order_ids_and_invalid_reference_date():
    df = pd.DataFrame({
        "Customer_ID": ["C1", "C2"],
        "Order_ID": ["O1", "O1"],
        "Order_Date": ["2025-01-01", "2025-01-02"],
        "Order_Value": [10, 20],
    })
    with pytest.raises(AnalysisError, match="Repeated order IDs"):
        calculate_rfm(df, suggest_column_mapping(df))

    df = transactions()
    with pytest.raises((ValueError, AnalysisError)):
        calculate_rfm(df, suggest_column_mapping(df), reference_date="not-a-date")


def test_revenue_concentration_refuses_negative_customer_net_revenue():
    df = pd.DataFrame({
        "Customer_ID": ["C1", "C2"],
        "Order_ID": ["O1", "O2"],
        "Order_Value": [-10, 100],
    })
    with pytest.raises(AnalysisError, match="negative net revenue"):
        calculate_revenue_concentration(df, suggest_column_mapping(df))


def test_result_validation_rejects_infinite_summary_and_table_values():
    result = AnalysisResult("CA-X", "test", {"score": float("inf")}, pd.DataFrame({"x": [1.0, np.inf]}))
    problems = validate_result(result)
    assert any("not finite" in p for p in problems)
    assert any("infinite" in p for p in problems)


def test_csv_dialect_detection_and_xlsx_sheet_inventory():
    csv = b"Customer_ID;Order_Value\nC1;10\nC2;20\n"
    loaded = load_uploaded_file("semi.csv", csv)
    assert list(loaded[0].dataframe.columns) == ["Customer_ID", "Order_Value"]
    assert loaded[0].dataframe.shape == (2, 2)

    buff = BytesIO()
    with pd.ExcelWriter(buff, engine="openpyxl") as writer:
        pd.DataFrame({"Customer_ID": ["C1"]}).to_excel(writer, sheet_name="Customers", index=False)
        pd.DataFrame({"Order_ID": ["O1"]}).to_excel(writer, sheet_name="Orders", index=False)
    workbook = load_uploaded_file("multi.xlsx", buff.getvalue())
    assert [t.table_name for t in workbook] == ["Customers", "Orders"]


def test_ingestion_rejects_empty_malformed_and_wrong_extension_files():
    with pytest.raises(IngestionError):
        load_uploaded_file("empty.csv", b"")
    with pytest.raises(IngestionError, match="valid ZIP-based Excel"):
        load_uploaded_file("bad.xlsx", b"not-an-xlsx")
    with pytest.raises(IngestionError, match="Unsupported"):
        load_uploaded_file("data.json", b"{}")


def test_profile_reports_invalid_dates_and_duplicate_rows_without_cleaning():
    df = pd.DataFrame({
        "Order_Date": ["2025-01-01", "bad-date", None],
        "Revenue": [10.0, 10.0, np.nan],
    })
    original = df.copy(deep=True)
    profile = profile_dataframe(df)
    date_col = next(c for c in profile["columns"] if c["column"] == "Order_Date")
    assert date_col["date_parse_failure_count"] == 1
    pd.testing.assert_frame_equal(df, original)


def test_csv_export_round_trip_preserves_values_and_unicode():
    original = pd.DataFrame({"Customer_ID": ["客户-1", "C2"], "Revenue": [10.5, 20.0]})
    payload = dataframe_to_csv_bytes(original)
    restored = pd.read_csv(BytesIO(payload), encoding="utf-8-sig")
    pd.testing.assert_frame_equal(restored, original)


def test_deterministic_interpretation_does_not_claim_llm_was_used():
    result = calculate_rfm(transactions(), suggest_column_mapping(transactions()), reference_date="2025-03-01")
    narrative = deterministic_interpretation(result)
    assert narrative["narrative_provider_used"] is False
    assert narrative["analysis_id"] == "CA-017"
    assert "narrative_provider_used" not in " ".join(narrative["interpretation_lines"])


def test_registry_has_unique_ids_and_every_status_is_valid():
    df = transactions()
    mapping = suggest_column_mapping(df)
    assessment = assess_dataset(df)
    ids = [item["analysis_id"] for item in assessment.feasibility]
    assert len(ids) == 35
    assert len(ids) == len(set(ids))
    assert all(item["feasibility_status"] in VALID_STATUSES for item in assessment.feasibility)


def test_result_csv_does_not_include_dataframe_index():
    df = pd.DataFrame({"metric": [1, 2]}, index=[9, 10])
    decoded = dataframe_to_csv_bytes(df).decode("utf-8-sig")
    assert decoded.splitlines()[0] == "metric"
    assert "9," not in decoded


def test_nps_feasibility_rejects_out_of_range_scores_and_flags_mixed_scores():
    all_bad = pd.DataFrame({"NPS_Score": [11, -1, "invalid"]})
    mapping_bad = suggest_column_mapping(all_bad)
    result_bad = next(
        row for row in assess_feasibility(all_bad, mapping_bad, confirmed_definitions={"nps_definition"})
        if row["analysis_id"] == "CA-011"
    )
    assert result_bad["feasibility_status"] == "Not feasible"
    assert "0–10" in result_bad["blockers"]

    mixed = pd.DataFrame({"NPS_Score": [9, 10, 11, "invalid"]})
    result_mixed = next(
        row for row in assess_feasibility(
            mixed, suggest_column_mapping(mixed), confirmed_definitions={"nps_definition"}
        ) if row["analysis_id"] == "CA-011"
    )
    assert result_mixed["feasibility_status"] == "Partially feasible"
    assert "must be excluded or reviewed" in result_mixed["warnings"]


def test_feasibility_flags_malformed_required_numeric_and_date_fields():
    df = pd.DataFrame({
        "Customer_ID": ["C1", "C2", "C3"],
        "Order_ID": ["O1", "O2", "O3"],
        "Order_Date": ["2025-01-01", "not-a-date", "2025-01-03"],
        "Order_Value": [10, "not-money", 30],
    })
    definitions = {"rfm_definition"}
    rows = assess_feasibility(df, suggest_column_mapping(df), confirmed_definitions=definitions)
    rfm = next(row for row in rows if row["analysis_id"] == "CA-017")
    assert rfm["feasibility_status"] == "Partially feasible"
    assert "non-numeric" in rfm["warnings"]
    assert "unparseable date" in rfm["warnings"]


def test_no_parseable_required_dates_requires_clarification_not_false_feasibility():
    df = pd.DataFrame({
        "Customer_ID": ["C1", "C2"],
        "Order_ID": ["O1", "O2"],
        "Order_Date": ["bad-date", "also-bad"],
        "Order_Value": [10, 20],
    })
    row = next(
        item for item in assess_feasibility(df, suggest_column_mapping(df), confirmed_definitions={"rfm_definition"})
        if item["analysis_id"] == "CA-017"
    )
    assert row["feasibility_status"] == "Needs clarification"
    assert "no parseable dates" in row["blockers"]


def test_revenue_concentration_known_answer_metrics():
    result = calculate_revenue_concentration(transactions(), suggest_column_mapping(transactions()))
    assert result.summary["total_historical_revenue"] == pytest.approx(250.0)
    assert result.summary["customers_analyzed"] == 3
    assert result.summary["top_10pct_customer_count"] == 1
    assert result.summary["top_10pct_customer_revenue_share_pct"] == pytest.approx(60.0)
    assert result.summary["hhi_on_customer_revenue_shares"] == pytest.approx(0.4688)
    assert result.table["Revenue_Share_Pct"].sum() == pytest.approx(100.0)


def test_rfm_recency_score_direction_is_correct():
    df = pd.DataFrame({
        "Customer_ID": ["old", "recent", "middle"],
        "Order_ID": ["O1", "O2", "O3"],
        "Order_Date": ["2025-01-01", "2025-03-01", "2025-02-01"],
        "Order_Value": [10, 10, 10],
    })
    result = calculate_rfm(df, suggest_column_mapping(df), reference_date="2025-04-01")
    lookup = result.table.set_index("Customer_ID")
    assert lookup.loc["recent", "R_Score"] > lookup.loc["middle", "R_Score"]
    assert lookup.loc["middle", "R_Score"] > lookup.loc["old", "R_Score"]


def test_complete_supported_analysis_batch_matches_known_totals():
    df = transactions()
    mapping = suggest_column_mapping(df)
    feasibility = assess_feasibility(
        df, mapping,
        confirmed_definitions={"rfm_definition", "value_definition", "order_granularity", "repeat_definition", "period_definition", "profitability_definition"},
    )
    approved = ["CA-017", "CA-018", "CA-019", "CA-020", "CA-030"]
    results, errors = execute_approved_analyses(df, mapping, feasibility, approved)
    assert not errors
    by_id = {r.analysis_id: r for r in results}
    assert set(by_id) == set(approved)
    assert by_id["CA-018"].summary["total_historical_revenue"] == pytest.approx(250.0)
    assert by_id["CA-019"].summary["average_order_value"] == pytest.approx(62.5)
    assert by_id["CA-020"].summary["repeat_purchase_rate_pct"] == pytest.approx(100 / 3, abs=0.01)
    assert by_id["CA-030"].summary["top_10pct_customer_revenue_share_pct"] == pytest.approx(60.0)
    assert all(validate_result(result) == [] for result in results)


def test_result_validator_checks_nested_numpy_nonfinite_values():
    result = AnalysisResult("CA-X", "nested", {"metadata": {"scores": [np.float64(np.nan)]}})
    issues = validate_result(result)
    assert any("summary.metadata.scores[0]" in issue for issue in issues)


def test_xlsx_expanded_content_limit_is_enforced(monkeypatch):
    buff = BytesIO()
    with pd.ExcelWriter(buff, engine="openpyxl") as writer:
        pd.DataFrame({"x": [1]}).to_excel(writer, index=False)
    import customer_analytics.ingestion as ingestion
    monkeypatch.setattr(ingestion, "MAX_EXCEL_UNCOMPRESSED_MB", 0)
    with pytest.raises(IngestionError, match="uncompressed-content limit"):
        load_uploaded_file("tiny.xlsx", buff.getvalue())


def test_zip_valid_but_not_excel_workbook_returns_controlled_ingestion_error():
    import zipfile
    payload = BytesIO()
    with zipfile.ZipFile(payload, "w") as archive:
        archive.writestr("not-an-excel-file.txt", "hello")
    with pytest.raises(IngestionError, match="Could not read Excel workbook"):
        load_uploaded_file("not-excel.xlsx", payload.getvalue())


def test_all_invalid_revenue_rows_fail_with_clear_message():
    df = pd.DataFrame({"Customer_ID": ["C1", None], "Order_ID": ["O1", "O2"], "Order_Value": ["bad", None]})
    with pytest.raises(AnalysisError, match="No valid customer IDs and numeric revenue"):
        calculate_customer_revenue(df, suggest_column_mapping(df))
