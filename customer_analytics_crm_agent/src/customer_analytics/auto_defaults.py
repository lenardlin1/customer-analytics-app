"""Data-derived suggestions for Customer Analytics controls.

These are recommendations, not inferred business truth. Each suggestion includes
an explanation so the UI can expose when data is insufficient or assumptions apply.
"""
from __future__ import annotations

import math
import re
from typing import Any

import numpy as np
import pandas as pd


def _norm(text: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text).casefold()).strip()


def _mapped_column(df: pd.DataFrame, mapping: dict[str, str] | None, key: str) -> str | None:
    value = (mapping or {}).get(key)
    return value if value in df.columns else None


def _round_to_step(value: float, step: int = 5) -> int:
    return int(step * round(float(value) / step))


def recommend_inactivity_window(df: pd.DataFrame, mapping: dict[str, str] | None = None) -> tuple[int, str]:
    """Suggest inactivity days from observed repeat-purchase intervals.

    When at least five positive intervals exist, uses their 75th percentile,
    rounded to 5-day increments and bounded to 7–365 days. With a date span but
    too few repeats, uses 25% of the covered date span and explicitly labels it
    as a fallback heuristic. Without usable dates, retains a transparent 90-day
    starting point rather than pretending to infer a business rule.
    """
    date_col = _mapped_column(df, mapping, "txn_date")
    customer_col = _mapped_column(df, mapping, "customer_id")
    if date_col is None:
        return 90, "No usable transaction-date mapping was identified. Using a 90-day starting point; confirm the business inactivity definition."

    dates = pd.to_datetime(df[date_col], errors="coerce", format="mixed")
    valid = dates.dropna()
    if valid.empty:
        return 90, "The mapped date column contains no parseable dates. Using a 90-day starting point; fix date parsing before interpreting inactivity."

    span = int((valid.max() - valid.min()).total_seconds() // 86400)
    if customer_col is not None:
        events = pd.DataFrame({"customer": df[customer_col], "date": dates}).dropna()
        events = events.sort_values(["customer", "date"])
        gaps = events.groupby("customer", sort=False)["date"].diff().dt.total_seconds().div(86400)
        gaps = gaps[(gaps > 0) & gaps.notna()]
        if len(gaps) >= 5:
            raw = float(gaps.quantile(0.75))
            suggested = int(np.clip(_round_to_step(raw), 7, 365))
            return suggested, (
                f"Auto-suggested from {len(gaps):,} positive within-customer repeat-activity gaps: "
                f"the 75th percentile is {raw:.1f} days, rounded to a 5-day step and bounded to 7–365 days. "
                "This is an inactivity proxy, not a confirmed churn definition."
            )

    if span >= 7:
        raw = max(7.0, span * 0.25)
        suggested = int(np.clip(_round_to_step(raw), 7, 365))
        return suggested, (
            f"Only {span:,} days of date coverage and insufficient repeat gaps were available. "
            f"Using a fallback heuristic of 25% of the observed date span ({raw:.1f} days), rounded and bounded to 7–365 days. "
            "Treat this as a starting point and confirm the business rule."
        )
    return 7, (
        f"The observed date range spans only {span} day(s), which is too short to estimate a reliable inactivity rule. "
        "Seven days is a minimum technical suggestion; confirm the business definition or upload a longer history."
    )


def recommend_csat_threshold(df: pd.DataFrame, mapping: dict[str, str] | None = None) -> tuple[float, str]:
    """Suggest a satisfied threshold based on the observed satisfaction scale."""
    score_col = _mapped_column(df, mapping, "csat_score")
    if score_col is None:
        return 4.0, "No CSAT/satisfaction-score column was confidently mapped. The 4.0 default is only a placeholder for a common 1–5 scale; map the correct column and confirm its scale."

    scores = pd.to_numeric(df[score_col], errors="coerce")
    scores = scores.replace([np.inf, -np.inf], np.nan).dropna()
    if scores.empty:
        return 4.0, f"The mapped CSAT column '{score_col}' has no usable numeric values. The 4.0 default is a placeholder; verify encoding and scale."

    minimum, maximum = float(scores.min()), float(scores.max())
    unique = set(scores.unique().tolist())
    if unique.issubset({0.0, 1.0}) and len(unique) <= 2:
        threshold = 1.0
        scale = "binary 0/1"
    elif minimum >= 0 and maximum <= 5:
        threshold = 4.0
        scale = "likely 0–5 or 1–5 satisfaction scale"
    elif minimum >= 0 and maximum <= 7 and maximum > 5:
        threshold = 6.0
        scale = "likely 1–7 satisfaction scale"
    elif minimum >= 0 and maximum <= 10 and maximum > 7:
        threshold = 8.0
        scale = "likely 0–10 or 1–10 satisfaction scale"
    elif minimum >= 0 and maximum <= 100 and maximum > 10:
        threshold = 80.0
        scale = "likely 0–100 satisfaction scale"
    else:
        threshold = float(math.ceil((0.8 * maximum) * 2) / 2) if maximum > 0 else 0.0
        scale = "non-standard observed score range"

    return threshold, (
        f"Auto-suggested {threshold:g} using a {scale}; the uploaded values range from {minimum:g} to {maximum:g} "
        f"across {len(scores):,} numeric responses. Confirm the original survey's allowed scale and satisfaction rule; "
        "observed minimum/maximum may not reveal the full questionnaire scale."
    )


def recommend_cluster_count(df: pd.DataFrame, mapping: dict[str, str] | None = None, max_clusters: int = 8) -> tuple[int, str]:
    """Pick k by maximum silhouette score when the dataset supports clustering.

    Mirrors the app's general numeric-feature approach: numeric columns are used,
    likely identifiers are excluded, and repeated customer rows are aggregated to
    customer-level averages when a customer key is mapped.
    """
    id_tokens = {"id", "phone", "zip", "postal", "postcode", "customer number", "order number", "account number"}
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                    if not any(token in _norm(c).split() for token in id_tokens)]
    # Drop constant/all-missing features before imputing or clustering.
    numeric_cols = [c for c in numeric_cols if df[c].notna().any() and df[c].nunique(dropna=True) > 1]
    cust_col = _mapped_column(df, mapping, "customer_id")

    if cust_col is not None and numeric_cols:
        feature_frame = df[[cust_col] + numeric_cols].copy()
        feature_frame[numeric_cols] = feature_frame[numeric_cols].apply(pd.to_numeric, errors="coerce")
        base = feature_frame.groupby(cust_col, dropna=True)[numeric_cols].mean(numeric_only=True)
    elif numeric_cols:
        base = df[numeric_cols].apply(pd.to_numeric, errors="coerce").copy()
    else:
        return 3, "No suitable varying numeric features were detected for exploratory clustering. Three is only a placeholder; map/convert meaningful numeric behaviour fields before interpreting segments."

    base = base.replace([np.inf, -np.inf], np.nan)
    base = base.dropna(axis=1, how="all")
    if base.shape[0] > 1500:
        base = base.sample(n=1500, random_state=42)
    if base.shape[0] < 6 or base.shape[1] < 2:
        fallback = max(2, min(3, int(base.shape[0]) - 1)) if base.shape[0] >= 3 else 2
        return fallback, (
            f"Detected {base.shape[0]} eligible customer/row records and {base.shape[1]} varying numeric feature(s). "
            "There are too few records or dimensions to select cluster count reliably with silhouette scoring; "
            f"using {fallback} as an editable fallback."
        )

    try:
        from sklearn.cluster import KMeans
        from sklearn.impute import SimpleImputer
        from sklearn.metrics import silhouette_score
        from sklearn.preprocessing import StandardScaler

        imputed = SimpleImputer(strategy="median").fit_transform(base)
        scaled = StandardScaler().fit_transform(imputed)
        n = scaled.shape[0]
        upper_k = min(max(2, int(max_clusters)), 8, n - 1)
        scores: list[tuple[float, int]] = []
        for k in range(2, upper_k + 1):
            labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(scaled)
            n_labels = len(set(labels))
            if n_labels < 2 or n_labels >= n:
                continue
            score = float(silhouette_score(scaled, labels, sample_size=min(n, 1000), random_state=42))
            if np.isfinite(score):
                scores.append((score, k))
        if not scores:
            return 3, "The available numeric data did not produce a valid silhouette score across the candidate cluster counts. Three is an editable fallback; inspect the feature selection and data quality."
        # Highest score; in a tie, prefer fewer clusters.
        best_score, best_k = sorted(scores, key=lambda item: (-item[0], item[1]))[0]
        comparison = ", ".join(f"k={k}: {score:.3f}" for score, k in sorted(scores, key=lambda item: item[1]))
        return int(best_k), (
            f"Auto-selected {best_k} clusters using the highest silhouette score from {base.shape[0]:,} eligible customer/row records "
            f"and {base.shape[1]} varying numeric features. Candidate scores: {comparison}. "
            "This is an exploratory statistical suggestion, not proof that the segments are commercially meaningful."
        )
    except Exception as exc:
        return 3, f"Automatic silhouette selection could not complete ({type(exc).__name__}: {exc}). Three is an editable fallback; review numeric feature quality and dependencies."


def recommend_analysis_defaults(df: pd.DataFrame, mapping: dict[str, str] | None = None) -> dict[str, Any]:
    inactivity, inactivity_note = recommend_inactivity_window(df, mapping)
    csat, csat_note = recommend_csat_threshold(df, mapping)
    clusters, clusters_note = recommend_cluster_count(df, mapping)
    return {
        "inactivity_days": inactivity,
        "inactivity_note": inactivity_note,
        "csat_threshold": csat,
        "csat_note": csat_note,
        "n_clusters": clusters,
        "clusters_note": clusters_note,
    }
