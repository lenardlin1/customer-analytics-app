from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from customer_analytics.auto_defaults import (
    recommend_analysis_defaults,
    recommend_csat_threshold,
    recommend_inactivity_window,
    recommend_cluster_count,
)
from customer_analytics.full_analytics import FIELD_DEFS, NOT_MAPPED, guess_column


def _mapping(df):
    return {key: guess_column(df, aliases) for key, (_, aliases) in FIELD_DEFS.items()}


def test_inactivity_window_uses_observed_customer_repeat_gaps():
    df = pd.DataFrame({
        "Customer_ID": ["A", "A", "A", "B", "B", "B", "C", "C", "C"],
        "Transaction_Date": ["2026-01-01", "2026-01-11", "2026-02-10",
                             "2026-01-01", "2026-01-21", "2026-03-22",
                             "2026-01-01", "2026-01-16", "2026-02-15"],
    })
    days, note = recommend_inactivity_window(df, _mapping(df))
    assert 7 <= days <= 365
    assert "positive within-customer repeat-activity gaps" in note


def test_inactivity_fallback_discloses_coverage_heuristic():
    df = pd.DataFrame({"Customer_ID": ["A", "B"], "Transaction_Date": ["2026-01-01", "2026-05-01"]})
    days, note = recommend_inactivity_window(df, _mapping(df))
    assert days == 30
    assert "fallback heuristic" in note


def test_csat_satisfaction_scale_five_suggests_four():
    df = pd.DataFrame({"CSAT_Score": [1, 2, 3, 4, 5, None]})
    threshold, note = recommend_csat_threshold(df, _mapping(df))
    assert threshold == 4.0
    assert "Confirm the original survey" in note


def test_csat_scale_ten_suggests_eight():
    df = pd.DataFrame({"CSAT_Score": [0, 2, 5, 8, 10]})
    assert recommend_csat_threshold(df, _mapping(df))[0] == 8.0


def test_no_csat_mapping_does_not_claim_data_driven_threshold():
    df = pd.DataFrame({"Mystery": [1, 2, 3]})
    mapping = {key: NOT_MAPPED for key in FIELD_DEFS}
    threshold, note = recommend_csat_threshold(df, mapping)
    assert threshold == 4.0
    assert "only a placeholder" in note


def test_cluster_count_uses_silhouette_scores_on_suitable_data():
    rng = np.random.default_rng(19)
    a = rng.normal(loc=0.0, scale=0.25, size=(35, 2))
    b = rng.normal(loc=4.0, scale=0.25, size=(35, 2))
    c = rng.normal(loc=8.0, scale=0.25, size=(35, 2))
    df = pd.DataFrame(np.vstack([a, b, c]), columns=["Spend", "Orders"])
    k, note = recommend_cluster_count(df, {key: NOT_MAPPED for key in FIELD_DEFS})
    assert 2 <= k <= 8
    assert "silhouette score" in note
    assert "Candidate scores" in note


def test_recommendations_respond_to_each_new_dataset():
    df = pd.read_csv(ROOT / "data" / "synthetic_customer_transactions.csv")
    result = recommend_analysis_defaults(df, _mapping(df))
    assert result["inactivity_days"] != 90
    assert result["csat_threshold"] == 4.0
    assert 2 <= result["n_clusters"] <= 8
    assert "inactivity_note" in result and "csat_note" in result and "clusters_note" in result
