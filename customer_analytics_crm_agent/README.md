# Customer Analytics & CRM Intelligence Agent

A local Streamlit academic prototype that combines an evidence-aware CRM capability knowledge base, comparison and gap assessment with customer-data profiling, analysis feasibility, a broad set of deterministic customer metrics, automatic interpretation, and exportable reports.

## What is implemented

- CRM evidence matrix: 84 capability/platform pairs across Zoho CRM, HubSpot, Salesforce, monday CRM, Pipedrive and ActiveCampaign. Sixteen records have feature-specific official documentation reviewed for this curated build; the remaining records are explicitly **Not verified**, not unavailable.
- Capability search and selected-platform comparison, with official source links, plan caveats, verification dates and confidence tags.
- Gap assessment with evidence-aware classifications and adjustable 1–5 prioritization ratings. Unverified features cannot become confirmed missing-feature findings.
- CSV/XLSX/XLSM upload and worksheet selection; optional `.xls` support through `xlrd`.
- Suggested field mappings that must be reviewed by the user.
- **Dataset-driven analysis defaults:** the inactivity proxy window is estimated from observed within-customer repeat-activity gaps where enough timestamps are available; the CSAT threshold is suggested from the observed response scale; and exploratory cluster count is selected using silhouette scoring when at least two varying numeric features and sufficient records are available. Each suggestion includes its method/fallback note and remains editable. When the relevant column is not detected, the app labels the result as a fallback instead of claiming data-driven inference. New uploads/worksheets receive fresh setting defaults.
- Missingness/schema profiling, IQR-based outlier flags, row-level outlier review queues with selected treatment/action status, exact duplicate reporting with source row numbers, and opt-in documented treatments. The uploaded source is copied and not overwritten by cleaning. The Excel audit workbook includes a data dictionary, duplicate/outlier audits, cleaning log, feasibility, calculated result tables and interpretations.
- Analysis feasibility checks and a deterministic analytics engine for CSAT, observed churn or an explicit inactivity proxy, retention/cohorts, historical customer value, AOV when row grain is confirmed, event journey sequences, exploratory segmentation, text sentiment screening, customer behavior, loyalty/repeat purchase, CES, NPS, CAC if inputs align, historical-value:CAC ratio, FCR, ticket response/resolution time, descriptive price/quantity association and RFM. Extra descriptive statistics and channel summaries are available when supported.
- Optional single-test statistical workflow for Pearson correlation, one-predictor OLS regression and chi-square independence tests, with sample-size/assumption notes and automatic interpretation.
- Optional supervised churn-model comparison (logistic regression vs random forest) only when the user maps an explicit binary churn outcome, selects pre-outcome predictors and confirms the feature timing. A stratified holdout evaluation includes ROC AUC, accuracy/balanced accuracy, precision, recall, F1, confusion matrices, top coefficients/importances and holdout predictions. It is exploratory, not deployment-ready.
- Automatically created rule-based interpretation from the result summaries/tables, including findings, implications, proposed actions, KPI ideas and limitations. Purpose-driven Plotly visualizations are generated from the actual cohort, survey, segment, channel, RFM, model-evaluation or statistical result tables when the format is supported.
- Markdown/HTML/CSV/Excel/JSON exports.
- Optional OpenAI narration from aggregated interpretation summaries only; no LLM call is made by default and calculations never depend on it.
- Fictional customer transaction dataset with 602 rows, 140 customers, one exact duplicate row and documented missingness. It is synthetic and must not be represented as real business evidence.

## Windows setup

Python 3.11–3.13 is recommended for a conservative dependency environment. In PowerShell:

```powershell
cd customer_analytics_crm_agent
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
$env:PYTHONPATH = "src"
pytest -q
python -m streamlit run app.py
```


### Windows Command Prompt (CMD)

If you use CMD rather than PowerShell, activate and run using these commands (do not use `$env:PYTHONPATH`, which is PowerShell-only):

```bat
cd /d "C:\path\to\customer_analytics_crm_agent"
set PYTHONPATH=src
python -m pytest -q
python -m python -m streamlit run app.py
```

The three analysis settings are calculated when a dataset/worksheet is loaded. If you manually override the inferred column mappings, confirm the explanatory notes and verify that the auto-detected columns are appropriate before interpreting the analysis.

## macOS / Linux setup

```bash
cd customer_analytics_crm_agent
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
export PYTHONPATH=src
pytest -q
python -m streamlit run app.py
```

Streamlit prints a local URL, usually `http://localhost:8501`.

## Predictive churn-model demonstration data

`data/synthetic_churn_modeling.csv` contains 320 fictional customer-level rows. `Churn_Within_Next_90_Days` is a synthetic outcome for a 90-day window after a fictional baseline snapshot; the numerical/categorical predictors are intended to represent baseline-period fields. The app requires an explicit mapping and a user confirmation that selected predictors predate the outcome. Model metrics from this file are reproducibility examples only, not a claim about a real population.

## Optional AI narration

Local deterministic calculations and interpretation work without an API key. To enable extra language-model narration, first install the optional dependency with `pip install -r requirements-llm.txt`, then set `OPENAI_API_KEY` and optionally `OPENAI_MODEL` in your process environment. Then open **AI Insights & Recommendations** and explicitly click the optional narration button. Only aggregate interpretations are sent; raw rows, customer identifiers and feedback text are not sent by the adapter. External processing must still be reviewed against the organisation's data policy.

## Suggested professor demonstration

1. Open Overview and download/show the synthetic dataset.
2. In CRM Intelligence, search for “predict customer churn”. Show the Zoho customer churn documentation and HubSpot health-score plan caveat. Explain that those native capabilities mean the project's claim is not that CRMs universally lack churn prediction.
3. In Capability Comparison, compare customer health/churn, workflow automation and sales reporting across six vendors. Point out the 68 not-verified records as explicit research tasks, not product deficiencies.
4. In Gap Assessment, choose Zoho CRM → customer health/churn or risk-based retention task routing; record the actual configuration as unknown. Show how the engine treats unverified account details as a research task, not a product gap. Adjust the prioritization weights and show that the score is an explicit stakeholder heuristic.
5. In Customer Analytics, upload `data/synthetic_customer_transactions.csv`. Confirm mappings for Customer ID, Order ID, transaction date, revenue, quantity, price, CSAT, CES, NPS, text feedback, channel and product. Use “Flag only” first, then show how exact-duplicate retention affects row-grain-sensitive AOV.
6. Show Data Quality: one exact duplicate row, missingness, summary outlier ranges and the row-level outlier audit. Explain source-row traceability and why an outlier flag is not proof of an error.
7. Open AI Insights & Recommendations. Show actual NPS, RFM, retention/inactivity proxy, sentiment and historical value, with limitations. Demonstrate that CAC is not calculated without aligned acquisition spend and new-customer counts.
8. To demonstrate supervised models, upload `data/synthetic_churn_modeling.csv`, map `Customer_ID` and `Churn_Within_Next_90_Days`, use the six baseline predictors, select `Churn` as the positive class and confirm predictors predate the outcome window. Compare logistic regression vs random forest on the holdout; discuss recall/precision trade-offs, not accuracy alone. These records are synthetic and do not establish real customer risk.
9. Run one optional statistical test if relevant, then export the Markdown/HTML and full Excel audit workbook. Conclude with a pilot measurement plan rather than an invented ROI claim.

## Research sources used in the first evidence pass

- Zoho CRM Workflow Rules: https://help.zoho.com/portal/en/kb/crm/automate-business-processes/workflows/articles/configuring-workflow-rules
- Zoho CRM customer churn using Zia: https://help.zoho.com/portal/en/kb/crm/zia-artificial-intelligence/prediction/articles/predict-customer-churn-using-zia
- HubSpot customer health score: https://knowledge.hubspot.com/help-desk/customize-a-health-score-in-the-customer-success-workspace
- HubSpot workflows and subscriptions: https://knowledge.hubspot.com/get-started/automate-your-processes
- Salesforce Sales Cloud Einstein / Sales Analytics: https://help.salesforce.com/s/articleView?id=sf.einstein_sales.htm
- Salesforce Einstein Forecasting: https://help.salesforce.com/s/articleView?id=sf.einstein_sales_forecasting_adjustments.htm&language=en_US&type=5
- Salesforce Einstein Prediction Builder retirement notice (published 2026-10-03): https://help.salesforce.com/s/articleView?id=004980243&language=en_US&type=1
- Salesforce prediction score workflows: https://help.salesforce.com/s/articleView?id=sales.custom_ai_prediction_builder_business_workflow.htm&language=en_US&type=5
- monday CRM dashboards: https://support.monday.com/hc/en-us/articles/4409464697618-Get-started-with-monday-CRM
- monday automations: https://support.monday.com/hc/en-us/articles/360001222900-Get-started-with-monday-automations
- Pipedrive Insights: https://support.pipedrive.com/en/article/insights-feature
- Pipedrive automation limits: https://support.pipedrive.com/en/article/automation-limits
- ActiveCampaign Predictive Sending: https://help.activecampaign.com/hc/en-us/articles/20315252350876-How-to-use-predictive-sending-to-automate-send-times
- ActiveCampaign deal automations: https://help.activecampaign.com/hc/en-us/articles/206821090-Using-automations-to-manage-your-deals

## Known limitations

- CRM research is a deliberately scoped inventory of 14 capability dimensions, not an exhaustive product feature catalogue. Most records remain unverified until their precise plan and workflow are researched.
- No live CRM OAuth/API connection or CRM write action is included. The app does not claim to change CRM records.
- Most customer metrics are descriptive and rule-based; clustering and keyword sentiment are exploratory. Predictive churn modeling is conditionally implemented for an explicit binary target and holdout evaluation, but is not production-validated or time-based. Predictive CLV, formal causal price elasticity, full marketing attribution, robust business-hour SLA/FCR reconstruction, production PDF output and a live OAuth/API connection are not implemented.
- Historical revenue/customer is not predictive CLV. An inactivity rule is a proxy unless an explicit churn definition is supplied. CAC/CLV:CAC need consistent cost, new-customer, cohort and period definitions.
- The prototype has no login, multiuser database, production retention policy, enterprise authorization or deployment security review. Run locally and avoid sensitive data unless approved safeguards are in place.
