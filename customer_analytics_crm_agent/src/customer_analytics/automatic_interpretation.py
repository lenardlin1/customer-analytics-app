"""Evidence-grounded automatic interpretation for the Customer Analytics Agent.

This module deliberately uses computed summaries and tables, not invented values.
It generates conservative rule-based business interpretations. It does not claim
causality from associations or present scenarios as observed financial outcomes.
"""
from __future__ import annotations
from typing import Any
import html
import pandas as pd


def _num(v: Any):
    try:
        x = float(v)
        return x if pd.notna(x) else None
    except (TypeError, ValueError):
        return None


def _table_facts(table: Any) -> list[str]:
    if not isinstance(table, pd.DataFrame) or table.empty:
        return []
    facts=[]
    # Report exact computed table values only; avoid asserting meanings from unknown columns.
    for col in table.columns:
        if pd.api.types.is_numeric_dtype(table[col]):
            s=pd.to_numeric(table[col], errors="coerce").dropna()
            if len(s):
                facts.append(f"{col}: min {s.min():,.3g}, max {s.max():,.3g} across {len(s)} usable value(s)")
    return facts[:4]


def interpret_analysis(name: str, result: dict[str, Any], dataset_rows: int | None = None) -> dict[str, Any]:
    """Create a structured interpretation from a real result object."""
    summary=str(result.get("summary") or "No computed summary was returned.").strip()
    note=str(result.get("note") or "").strip()
    key=name.casefold()
    findings=[summary]
    business="The result describes the uploaded data and the selected definitions. Business impact should be quantified only when suitable financial, population and time-period data are available."
    causes="The output is descriptive unless the method explicitly reports a validated inferential or predictive result. Any proposed explanation of why the pattern occurred is a hypothesis to investigate, not a proven cause."
    actions=["Review the underlying table and confirm that the mapped fields, population, time period and business definitions match the intended question."]
    kpis=["Track the same metric over a comparable future period and segment, using a consistent definition."]
    limits=[]
    if dataset_rows is not None:
        findings.insert(0, f"The analysis was run on an analysis dataset containing {dataset_rows:,} row(s); the metric-specific usable denominator may be smaller because of missing or invalid values.")
    if "predictive" in key and "churn" in key:
        business="The model comparison estimates how well selected pre-outcome customer attributes separate the explicitly defined churn class from the other observed class on a held-out sample. Use recall/precision and the costs of missed churn versus unnecessary outreach to choose an operating threshold; the default 0.5 threshold is not necessarily appropriate."
        actions=["Compare ROC AUC, balanced accuracy, recall and precision against a simple baseline and the cost of each error before selecting a model.","Review the holdout confusion matrix and the top features for data leakage, stability and business plausibility.","Before deployment, validate on a later time period, calibrate risk scores if probability estimates are required, and pilot a human-reviewed retention queue."]
        kpis=["Holdout ROC AUC and balanced accuracy","Recall and precision for the defined churn class","False-positive outreach cost and missed-churn cost","Model drift and calibration on a later period","Incremental retention versus a suitable holdout"]
        limits.append("This is an exploratory random holdout result, not a calibrated production probability or causal retention effect. Results can be optimistic if selected features contain post-outcome information or if a random split leaks time patterns.")
        if isinstance(result.get("table"),pd.DataFrame) and not result["table"].empty:
            metrics=result["table"]
            if "ROC AUC" in metrics.columns and "Model" in metrics.columns:
                best=metrics.sort_values("ROC AUC",ascending=False).iloc[0]
                findings.append(f"Highest held-out ROC AUC: {float(best['ROC AUC']):.3f} for {best['Model']}; this ranks discrimination on this sample, not real-world usefulness by itself.")
            if "Recall (positive class)" in metrics.columns:
                for _,row in metrics.iterrows():
                    findings.append(f"{row['Model']} positive-class recall: {float(row['Recall (positive class)']):.1%}; proportion of observed positive cases detected in the holdout at the tested threshold.")
    elif any(x in key for x in ["csat", "satisfaction"]):
        business="Satisfaction scores summarize reported customer experience; the distribution matters because an average can conceal dissatisfied subgroups. The threshold is meaningful only if it matches the survey scale and internal definition."
        actions=["Inspect low-score responses and segment them by channel, product, customer type or journey stage when those fields exist.","Review recurring complaints and test one targeted service improvement before scaling it."]
        kpis=["CSAT mean and distribution","Share below the defined satisfaction threshold","Repeat-contact or complaint rate, if available"]
        limits.append("Survey scores may be ordinal; averages should be supplemented with distributions and subgroup counts.")
    elif "churn" in key or "inactive" in key:
        business="A churn rate is meaningful only for a clearly defined eligible population and observation window. If the result is an inactivity proxy, it identifies customers who crossed a recency rule, not confirmed churn or a calibrated probability of leaving."
        actions=["Validate the churn definition, observation window and eligible denominator before comparing rates.","Prioritize outreach tests for high-value inactive customers if customer value and contact-permission data are available."]
        kpis=["Confirmed churn rate with fixed denominator","Reactivation rate","Retention by cohort/segment","Incremental contribution after intervention, if measurable"]
        limits.append("Inactivity thresholds are proxies unless confirmed churn labels and a defensible observation window exist.")
    elif "retention" in key or "cohort" in key:
        business="Cohort comparisons show how observed activity changes over time from each cohort's starting period. Compare like-for-like cohort ages; newer cohorts have less follow-up history."
        actions=["Compare cohorts at the same age and locate the first period with a material drop.","Design a lifecycle intervention just before the observed drop-off and compare against a holdout where feasible."]
        kpis=["Retention at fixed cohort ages","Repeat purchase rate","Incremental lift versus holdout"]
        limits.append("Purchase activity is not subscription retention; incomplete follow-up can make recent cohorts appear different.")
    elif "nps" in key or "net promoter" in key:
        business="NPS summarizes promoter share minus detractor share on the 0–10 recommendation scale. It is a perception indicator, not proof of future growth or actual referral behaviour."
        actions=["Review detractor feedback and compare response rates and NPS across relevant customer groups.","Close the loop on recurring issues and measure whether subsequent survey responses and behaviour improve."]
        kpis=["NPS","Promoter/passive/detractor proportions","Survey response rate","Repeat purchase/referral behaviour if linked"]
        limits.append("NPS is sensitive to sampling, response bias and survey timing; do not interpret it alone as a growth forecast.")
    elif "cac" in key or "acquisition cost" in key:
        business="Acquisition cost is comparable only when spend and acquired-customer counts cover the same channels, attribution rules and time period. A blended value can hide channel differences."
        actions=["Reconcile spend, attribution window and acquired-customer counts before reallocating budget.","Compare channel-level CAC with contribution-based customer value over a consistent horizon."]
        kpis=["CAC by channel","New customers acquired","Contribution margin from acquired customers","Payback period when data supports it"]
        limits.append("CAC is not reliable if spend attribution or acquired-customer counts are incomplete or mismatched.")
    elif "rfm" in key or "recency" in key or "monetary" in key:
        business="RFM-style groupings summarize recent purchasing, purchase frequency and monetary value. Segment labels depend on scoring rules and should be validated against actual future behaviour before they drive campaigns."
        actions=["Check segment sizes and value contribution, then target a small campaign matched to each segment's observed behaviour.","Use a holdout group to evaluate incremental repeat purchases or contribution."]
        kpis=["Segment size and revenue/contribution share","Repeat-purchase rate by segment","Campaign incremental lift"]
        limits.append("RFM is a descriptive segmentation, not a causal explanation or a guarantee of future value.")
    elif "sentiment" in key or "feedback" in key:
        business="Automated sentiment or theme labels are screening signals. They should be checked against a sample of original comments, especially for sarcasm, mixed sentiment, language differences and domain-specific wording."
        actions=["Manually review a stratified sample of positive, negative and uncertain comments.","Group recurring negative themes by product/channel and assign an operational owner for investigation."]
        kpis=["Validated sentiment precision on reviewed sample","Complaint theme frequency","Resolution time and repeat complaint rate"]
        limits.append("Model/lexicon sentiment is not a verified customer opinion until validated; text samples may not represent all customers.")
    elif "price" in key:
        business="Descriptive price and quantity patterns can identify associations, but they do not establish price elasticity when promotions, seasonality, product mix and customer selection are uncontrolled."
        actions=["Use a controlled price test or an appropriate causal design before changing prices based on this association.","Control for product, channel, promotion and time effects where data permits."]
        kpis=["Units and conversion by tested price","Contribution profit","Refund/return rate","Experiment confidence interval"]
        limits.append("Ordinary transaction correlation is not a causal estimate of price elasticity.")
    elif "correlation" in key:
        business="The computed coefficient describes a linear association between two measured variables. Use it to identify a pattern worth investigating, not to claim that changing one variable will change the other."
        actions=["Inspect a scatterplot and investigate influential observations or non-linear patterns.","Repeat the analysis within relevant segments or with a suitable multivariable/experimental design before changing policy."]
        kpis=["Correlation coefficient and confidence interval","Sample size and missingness","Outcome metric in a follow-up test"]
        limits.append("A statistically significant correlation can be small or confounded; it does not establish causation.")
    elif "regression" in key:
        business="Regression estimates the relationship between the selected outcome and predictor under the model assumptions. R-squared describes in-sample variation explained, not causal effect or guaranteed forecasting accuracy."
        actions=["Review residual diagnostics, influential cases and possible confounders.","If the goal is prediction, evaluate the model on held-out or time-separated data and compare with a baseline."]
        kpis=["Coefficient and confidence interval","R-squared and residual diagnostics","Out-of-sample error if used for prediction"]
        limits.append("Single-predictor regression is observational unless a suitable causal design supports stronger interpretation.")
    elif "chi-square" in key or "independence test" in key or "cross-tab" in key:
        business="The test evaluates whether two categorical variables appear statistically associated in the observed sample. The cross-tab shows where the counts differ; it does not explain why they differ."
        actions=["Review the contingency table and category counts, then assess whether the association is practically meaningful.","Check expected counts, independent observations and whether category pooling is defensible before acting."]
        kpis=["Within-group rates with denominators","Cramer's V or another effect size","Replicated association in a new period/sample"]
        limits.append("Low expected cell counts can weaken the chi-square approximation; association is not causal evidence.")
    elif "journey" in key or "conversion" in key or "event" in key:
        business="Event counts and stage transitions reveal where observed journeys concentrate or thin out. Conversion rates require a consistent eligible population at each stage and reliable event ordering."
        actions=["Validate event definitions, timestamps and stage denominators.","Investigate the largest supported drop-off and test a specific reduction in friction."]
        kpis=["Stage-to-stage conversion","Time between stages","Abandonment rate with defined denominator"]
        limits.append("Event data may omit offline steps or repeat events; counts are not conversion rates unless denominators are available.")
    elif "fcr" in key or "first contact" in key:
        business="FCR measures cases resolved at first contact only when the eligible-case denominator and first-contact definition are consistent."
        actions=["Audit a sample of cases to validate the FCR flag and check reopen/repeat-contact records.","Review low-FCR issue categories and improve knowledge articles or routing where evidence supports it."]
        kpis=["FCR with eligible-case denominator","Reopen rate","Repeat contact within defined window","CSAT after resolution"]
        limits.append("A row-level FCR flag can overstate resolution if cases or follow-up contacts are not linked correctly.")
    elif "resolution" in key or "ticket" in key or "response time" in key:
        business="Resolution-time summaries should be reviewed alongside the distribution and outliers; averages can be pulled upward by a small number of long-running cases. SLA breaches can be assessed only if a valid SLA target is available."
        actions=["Inspect the median and upper percentiles and review the longest cases by category and owner.","Compare SLA performance by ticket priority if SLA targets and priority data exist."]
        kpis=["Median and 90th percentile resolution time","SLA breach rate when target exists","Reopen rate","CSAT after resolution"]
        limits.append("SLA conclusions require explicit targets and correct business-hour/calendar rules.")
    elif "segment" in key or "cluster" in key:
        business="Segments or clusters summarize similarity in the selected features. Their commercial meaning depends on feature scaling, cluster stability and whether the groups differ on outcomes not used to create them."
        actions=["Profile each segment with interpretable feature summaries and customer counts.","Pilot segment-specific treatments and evaluate incremental outcomes rather than assuming labels imply needs."]
        kpis=["Segment size","Revenue/contribution per customer where available","Engagement and retention by segment","Incremental campaign lift"]
        limits.append("Exploratory clusters can be sensitive to scaling, feature selection, random initialization and chosen cluster count.")
    elif "clv" in key or "customer value" in key or "revenue" in key:
        business="Observed historical revenue/value is backward-looking and should not be presented as predictive customer lifetime value without a model and a stated horizon. Revenue also differs from profit or contribution."
        actions=["Use contribution profit and an explicit horizon for spending decisions where cost data exists.","If predictive CLV is required, validate a forward-looking model on a held-out time period."]
        kpis=["Historical revenue and contribution by segment","Repeat purchase and retention","Out-of-time CLV model error if a model is built"]
        limits.append("Historical revenue/customer is not predictive CLV or profit unless costs and a validated forecast are included.")
    elif "loyalty" in key or "repeat purchase" in key or "behaviour" in key or "behavior" in key:
        business="Repeat purchases and observed engagement describe behaviour; they do not independently prove attitudinal loyalty. Consider customer tenure, purchase opportunity and category buying cycles."
        actions=["Compare repeat behaviour across comparable tenure and acquisition cohorts.","Test relevant cross-sell or replenishment messages only where product and consent data support them."]
        kpis=["Repeat purchase rate","Time to next purchase","Cross-sell conversion","Contribution per customer"]
        limits.append("Behavioural proxies can be affected by tenure, purchase frequency and observation-window differences.")
    facts=_table_facts(result.get("table"))
    if facts:
        findings.append("Computed table ranges: " + "; ".join(facts) + ".")
    if note:
        limits.append(note)
    if not limits:
        limits.append("Interpretation is conditional on the source data, mappings, measurement definitions and row granularity being correct.")
    return {"analysis":name,"overview":f"This section evaluates {name.lower()} using the mapped fields and available records.","key_findings":findings,"business_implications":business,"root_cause_assessment":causes,"recommendations":actions,"kpis":kpis,"limitations":list(dict.fromkeys(limits)),"evidence_status":"Computed result + conservative rule-based interpretation; causal claims not inferred."}


def interpret_all(results: dict[str, dict[str, Any]], dataset_rows: int | None = None) -> dict[str, dict[str, Any]]:
    return {name: interpret_analysis(name, result, dataset_rows) for name,result in results.items()}


def render_interpretation_markdown(items: dict[str, dict[str, Any]]) -> str:
    out=["# Automatic Interpretation and Business Decision Intelligence", "", "Interpretations are generated from the computed result objects. Recommendations are proposals, not measured outcomes."]
    for item in items.values():
        out += ["", f"## {item['analysis']}", "", "### Analysis overview", item['overview'], "", "### Key findings"]
        out += [f"- {x}" for x in item['key_findings']]
        out += ["", "### Business implications", item['business_implications'], "", "### Root-cause assessment", item['root_cause_assessment'], "", "### Recommended actions"]
        out += [f"- {x}" for x in item['recommendations']]
        out += ["", "### KPIs to monitor"] + [f"- {x}" for x in item['kpis']]
        out += ["", "### Limitations and confidence notes"] + [f"- {x}" for x in item['limitations']]
    return "\n".join(out)


def render_html_report(title: str, report_text: str, interpretation_md: str) -> str:
    # Escape content; simple markdown-like headings/lists are retained as readable preformatted text.
    return "<!doctype html><html><head><meta charset='utf-8'><title>"+html.escape(title)+"</title><style>body{font-family:Arial,sans-serif;max-width:1000px;margin:36px auto;padding:0 20px;line-height:1.55;color:#202124}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f6f7f8;padding:18px;border-radius:8px}h1,h2{color:#19324d}</style></head><body><h1>"+html.escape(title)+"</h1><h2>Automatic interpretations</h2><pre>"+html.escape(interpretation_md)+"</pre><h2>Analysis report</h2><pre>"+html.escape(report_text)+"</pre></body></html>"
