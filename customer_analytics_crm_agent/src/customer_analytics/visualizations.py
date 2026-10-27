"""Purpose-driven charts for computed analytics outputs; no synthetic chart values are added."""
from __future__ import annotations
import re
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def build_analysis_chart(name: str, result: dict[str, Any]):
    table = result.get("table")
    if not isinstance(table, pd.DataFrame) or table.empty:
        return None
    key = name.casefold()
    df = table.copy()
    fig = None

    if "customer retention" in key and "Cohort month" in df.columns:
        month_cols = [c for c in df.columns if str(c).startswith("Month ")]
        if month_cols:
            matrix = df.set_index("Cohort month")[month_cols].apply(pd.to_numeric, errors="coerce")
            if not matrix.empty:
                fig = px.imshow(matrix, aspect="auto", color_continuous_scale="Blues",
                                labels={"x": "Months since cohort start", "y": "Cohort month", "color": "Active (%)"},
                                title="Observed customer cohort retention")
    elif "nps" in key and {"Group", "Count"}.issubset(df.columns):
        fig = px.bar(df, x="Group", y="Count", text="Percent", title="NPS response mix",
                     labels={"Group": "NPS category", "Count": "Responses", "Percent": "Share (%)"})
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    elif ("satisfaction" in key or "csat" in key or "ces" in key) and {"Score", "Responses"}.issubset(df.columns):
        fig = px.bar(df, x="Score", y="Responses", title=f"{name}: response distribution",
                     labels={"Score": "Recorded score", "Responses": "Response count"})
    elif "feedback sentiment" in key and {"Sentiment label", "Sentiment score"}.issubset(df.columns):
        counts = df["Sentiment label"].value_counts().rename_axis("Sentiment").reset_index(name="Responses")
        fig = px.bar(counts, x="Sentiment", y="Responses", title="Feedback sentiment screening counts",
                     labels={"Sentiment": "Lexicon label", "Responses": "Text responses"})
    elif "rfm" in key and {"Recency", "Frequency", "Monetary"}.issubset(df.columns):
        fig = px.scatter(df, x="Recency", y="Monetary", color="Frequency", hover_name="Customer ID" if "Customer ID" in df else None,
                         title="RFM customer profile", labels={"Recency": "Days since last purchase", "Monetary": "Observed monetary value", "Frequency": "Purchase frequency"})
    elif "customer segmentation" in key and "Segment" in df.columns:
        numeric = [c for c in df.select_dtypes(include="number").columns if df[c].nunique(dropna=True) > 1]
        if len(numeric) >= 2:
            fig = px.scatter(df, x=numeric[0], y=numeric[1], color="Segment",
                             title="Exploratory customer clusters", labels={numeric[0]: numeric[0], numeric[1]: numeric[1]})
    elif "customer behaviour" in key and {"records", "total_observed_amount"}.issubset(df.columns):
        fig = px.scatter(df, x="records", y="total_observed_amount", hover_name="Customer ID" if "Customer ID" in df else None,
                         title="Observed activity vs amount by customer", labels={"records": "Recorded activity rows", "total_observed_amount": "Total observed amount"})
    elif "channel performance" in key and {"channel", "total_amount"}.issubset(df.columns):
        fig = px.bar(df, x="channel", y="total_amount", title="Recorded revenue by channel",
                     labels={"channel": "Channel", "total_amount": "Recorded amount"})
    elif "channel mix" in key and {"Channel", "Rows"}.issubset(df.columns):
        fig = px.bar(df, x="Channel", y="Rows", title="Observed records by channel",
                     labels={"Channel": "Channel", "Rows": "Record count"})
    elif "average order value" in key and {"Order total"}.issubset(df.columns):
        fig = px.histogram(df, x="Order total", nbins=min(30, max(5, len(df)//3)), title="Order total distribution",
                           labels={"Order total": "Order total (source currency)"})
    elif "chi-square" in key or "independence test" in key:
        first = df.columns[0]
        numeric_cols = [c for c in df.columns[1:] if pd.to_numeric(df[c], errors="coerce").notna().sum() > 0]
        if numeric_cols and first in df:
            matrix = df.set_index(first)[numeric_cols].apply(pd.to_numeric, errors="coerce")
            if matrix.shape[0] and matrix.shape[1]:
                fig = px.imshow(matrix, aspect="auto", color_continuous_scale="Blues",
                                labels={"x": "Category", "y": str(first), "color": "Count"}, title="Observed contingency table")
    elif "predictive churn model evaluation" in key and {"Model", "ROC AUC", "Recall (positive class)", "Precision (positive class)"}.issubset(df.columns):
        metrics = df.melt(id_vars="Model", value_vars=["ROC AUC", "Recall (positive class)", "Precision (positive class)"],
                          var_name="Metric", value_name="Score")
        fig = px.bar(metrics, x="Model", y="Score", color="Metric", barmode="group", title="Churn model holdout metrics",
                     labels={"Score": "Holdout metric (0–1)", "Model": "Model"}, range_y=[0, 1])
    elif "descriptive statistics" in key and {"column", "mean"}.issubset({str(c).casefold() for c in df.columns}):
        lower = {str(c).casefold(): c for c in df.columns}
        col = lower.get("column")
        mean = lower.get("mean")
        if col and mean:
            fig = px.bar(df, x=col, y=mean, title="Numeric column means (descriptive only)", labels={col: "Column", mean: "Mean"})

    if fig is not None:
        fig.update_layout(template="plotly_white", font={"family": "Arial, sans-serif", "color": "#202124"},
                          margin={"l": 24, "r": 18, "t": 58, "b": 40}, legend_title_text="")
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(gridcolor="#e9ecef", zeroline=False)
    return fig
