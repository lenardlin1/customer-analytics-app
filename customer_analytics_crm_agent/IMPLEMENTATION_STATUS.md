# Implementation and Validation Status

**Project:** Customer Analytics & CRM Intelligence Agent  
**Validation date:** 27 October 2026  
**Status:** Runnable source package prepared; deterministic modules and exports validated. The Streamlit UI was syntax-checked but not launched in a browser from this build environment.

## 1. Existing code found and preserved

The saved project library contained earlier iterations of the customer's work, including a Phase 3 upload/profiling prototype, a Phase 9 starter, a Phase 10 analytics/validation project, and a separate automatic-interpretation iteration. This unified build preserves the Phase 10 ingestion, schema/registry, agent, reporting and regression-test foundations, and incorporates the broader analysis and interpretation functionality into a single modular project. It was an integration and extension rather than a claim that the complete requested application already existed.

## 2. Implementation status by module

| Module | Status | What is present | Limitations |
|---|---|---|---|
| File ingestion and mapping | Implemented | CSV and Excel workbook support, worksheet selection, schema profiling, suggested semantic mappings and user review | Unusual report-style spreadsheets with multi-row headers may need manual restructuring |
| Data quality and cleaning | Implemented for core checks | Missingness, types, exact duplicate candidates with source-row numbers, IQR outlier flags and row-level review, documented opt-in transformations, data dictionary and exportable audit workbook | Statistical flags do not decide business validity; fuzzy entity resolution is not fully implemented |
| Feasibility checks | Implemented | Evaluates available mappings, prerequisites and warnings before running supported metrics; refuses unsupported CAC and ambiguous order-level AOV cases | Definitions still need business owner confirmation for real deployments |
| Customer metrics | Implemented conditionally | CSAT, explicit churn or labelled inactivity proxy, retention/cohorts, historical customer value, conditional AOV, journey sequence summary, exploratory clustering, keyword sentiment, behavior, repeat-purchase proxy, CES, NPS, CAC when comparable inputs are confirmed, value-to-CAC comparison, FCR from an explicit flag, ticket elapsed-time metrics, descriptive price/quantity association and RFM | Metrics appear only when the fields and assumptions support them. Historical customer revenue is not predictive CLV; an inactivity rule is not a validated churn label; descriptive price association is not causal elasticity |
| Statistical tests | Implemented for selected methods | Pearson correlation, single-predictor OLS regression, and chi-square independence with suitability checks and automatic interpretation | Does not replace a full statistical workbench; assumption checks, multiple-testing control and advanced diagnostics remain limited |
| Predictive churn model | Exploratory, conditional | Logistic regression and random forest compared on a stratified holdout when there is an explicit binary target, adequate data, and user confirmation that selected features precede the outcome | No temporal backtest, threshold optimization, probability calibration, subgroup fairness audit or production deployment approval |
| Automatic interpretation | Implemented deterministically | Every supported result is linked to computed summaries/tables and gets key findings, business meaning, proposed actions/KPIs and limitations; unsupported computations are not narrated as successful | Rule-based interpretation is conservative and not a domain expert; some nuanced findings need human review |
| Optional LLM narration | Implemented as optional adapter | User-triggered OpenAI narration can summarize aggregate interpretation outputs; raw rows, customer IDs and feedback text are excluded from that adapter | Not executed in this validation because no API key/provider call was used. Requires optional dependency and account configuration |
| CRM evidence knowledge base | Implemented as curated first pass | 84 platform-capability records across six named CRM products; 16 records have feature-level official evidence and 68 are explicitly `not_verified`; URLs, scope/plan caveats, evidence status and verification dates are stored | Not an exhaustive inventory. An unverified record is not evidence of absence; tenant configuration/entitlements must still be checked |
| CRM comparison and gap assessment | Implemented | Search, selected-platform comparison, conservative gap classification, adjustable priority inputs, configurable weights and action-oriented report fields | No live tenant audit or direct product-plan API is connected; account configuration is user-supplied/unknown unless evidence is entered |
| Reports and exports | Implemented | Markdown, HTML, CSV, JSON and multi-sheet Excel with results, interpretations, data dictionary, feasibility and quality audits | Production PDF generation is not implemented |
| Streamlit interface | Source implemented; UI runtime not verified here | Unified navigation and workflow are present in `app.py`, with neutral styling, upload/mapping controls, result displays and exports | Streamlit was unavailable in the build runtime and public package installation was blocked by network access restrictions, so no live browser session was asserted |
| Live CRM integrations | Not implemented | Integration points are not misrepresented as active; the demo is evidence/search driven | No OAuth, CRM API connection, write actions, message sending, or tenant-specific feature introspection |
| Authentication/multi-user deployment | Not implemented | Local single-user prototype | No enterprise authentication, roles, hosted persistence controls or production security review |

## 3. Research scope and project positioning

The knowledge base covers 14 broad capability dimensions for Zoho CRM, HubSpot, Salesforce, monday CRM, Pipedrive and ActiveCampaign. It contains 84 platform-capability pairs: 16 with feature-level supporting evidence and 68 marked `not_verified`. That is a transparent research-coverage baseline, not a completeness claim.

The most defensible problem statement is **the evidence-to-decision gap between CRM vendor capabilities, a particular customer’s actual plan/configuration/data, and the business workflow required to act on customer insights**. It is not defensible to say that CRMs universally lack churn or health analytics. Some official vendor documentation describes churn prediction, health scoring, workflow automation or analytics. Product packaging and lifecycle can change; the knowledge base records a current Salesforce Einstein Prediction Builder retirement notice and cautions against assuming that a historically documented feature remains available indefinitely.

Potential gap classifications include functional capability, configuration, integration, data quality, analytics/decision support, process/adoption, subscription/cost, and research/evidence gaps. Findings about unknown tenant configuration or unverified features must remain unconfirmed until supporting evidence is supplied.

## 4. Automated test results

Command executed in the project directory:

```bash
python -m py_compile app.py src/customer_analytics/*.py tests/*.py
PYTHONPATH=src pytest -q
```

**Result: 73 tests passed in 3.91 seconds.** This includes the preserved Phase 10 regression suites and newer tests for CRM knowledge-base validation, conservative gap status handling, analytics calculations, feasible/infeasible cases, automatic interpretation grounding, data-quality audit exports, statistical testing, churn model safeguards and report outputs.

This is backend/unit and export validation. It is not proof that the Streamlit page ran in a browser; that UI runtime test remains outstanding in an environment with the declared dependencies installed.

## 5. Synthetic demonstration validation

The core transaction example is synthetic and contains 602 rows, 12 columns, 140 customers and one extra exact duplicate row. Profiling flagged 164 cell-level IQR outlier candidates; those are review flags, not 164 confirmed errors. The deterministic run produced 15 metric outputs and 15 corresponding interpretation objects. CAC was correctly not computed due to missing compatible acquisition-spend/new-customer inputs. AOV was refused because repeated order IDs conflicted with the dataset's declared order-level grain. A Pearson example was computed from 598 complete synthetic records; the association is not evidence of causation.

The second synthetic dataset has 320 rows and an explicit `Churn_Within_Next_90_Days` label. The test compares logistic regression and random forest with 80 holdout records. On this synthetic split, logistic regression produced ROC AUC about 0.771 and recall about 0.70 for the designated churn class; the random forest produced ROC AUC about 0.721 and recall about 0.45. These are reproducibility examples only—not expected performance for a real business and not production-ready risk scores.

Generated artifacts in `reports/` include the HTML, Markdown and Excel analysis report; a synthetic churn-model evaluation JSON; a CRM comparison CSV; a sample non-confirmed gap assessment JSON; and `validation_summary.json`.

## 6. Suggested professor demonstration (8–10 minutes)

1. **Problem and research framing (1 min):** State the evidence-to-decision gap. Explain that the prototype complements CRMs and does not assume existing native capabilities are absent.
2. **CRM evidence (1–2 min):** Search for customer churn/health and open source links. Show the plan caveat, verification date and 68 `not_verified` records; explain why `not_verified` is not `unavailable`.
3. **Gap assessment (1 min):** Enter a requirement such as prioritizing retention follow-ups. Keep the current tenant plan/configuration marked unknown. Change prioritization weights and show the reasoned next steps without presenting the demo as a live account audit.
4. **Data quality (1 min):** Load `data/synthetic_customer_transactions.csv`; show row/column counts, missingness, exact duplicate source rows and flagged outliers. Emphasize that the original input is preserved and flags are not automatically assumed erroneous.
5. **Automatic interpretation (1–2 min):** Run available analyses and open the interpretation report. Point to actual metric values, sample/denominator, action proposals, KPIs and limitations.
6. **Feasibility demonstration (1 min):** Show why CAC is not calculable without aligned acquisition spend and new-customer counts, and why order AOV is refused when row grain/order IDs conflict. Explain that refusal plus an alternative is valid system behavior.
7. **Predictive model (1–2 min):** Upload `data/synthetic_churn_modeling.csv`, map its explicit outcome and pre-outcome features, confirm timing and compare holdout metrics. Discuss recall/precision trade-offs and explain that random holdout results are not temporal validation.
8. **Report and business value (1 min):** Export the workbook and discuss a pilot plan: establish baselines, run a targeted intervention, measure holdout or pre/post outcomes, and avoid claiming causality or ROI before evidence exists.

## 7. Exact setup and run commands

### Windows PowerShell

```powershell
cd customer_analytics_crm_agent
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
$env:PYTHONPATH = "src"
pytest -q
streamlit run app.py
```

### macOS/Linux

```bash
cd customer_analytics_crm_agent
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
export PYTHONPATH=src
pytest -q
streamlit run app.py
```

Optional LLM narration requires `pip install -r requirements-llm.txt` plus environment variables `OPENAI_API_KEY` and optionally `OPENAI_MODEL`. It is not required for analytics or rule-based interpretations.

## 8. Highest-priority next improvements

1. Launch and test the actual Streamlit UI on the target Windows machine.
2. Verify more feature-level rows across every CRM and each applicable edition/plan, with a documented research query/date for each claim.
3. Add a read-only OAuth/API tenant inspector and real plan/configuration evidence, beginning with one CRM; keep writes human-approved and auditable.
4. Add temporal validation, calibration, threshold tuning and segment-level monitoring before any production churn-risk use.
5. Expand formally validated business definitions for revenue, margin, retention, churn, support SLA and attribution; add production-grade PDF if required.
6. Add login, authorization, encrypted persistent storage, deployment hardening and a reviewed privacy/retention policy before multi-user or sensitive-data use.


## Follow-up update — dataset-driven analysis settings (2026-10-27)

The Customer Analytics workspace now auto-suggests its three analysis parameters from the uploaded file and selected worksheet:

- **Inactivity proxy window:** uses the 75th percentile of observed positive within-customer repeat-activity intervals when there are enough intervals; otherwise applies a clearly disclosed date-span fallback or a 90-day placeholder when usable dates are absent.
- **Satisfied CSAT threshold:** maps likely rating scales to suggested conventions (binary 0/1 → 1; common 0/1–5 → 4; 1–7 → 6; 0/1–10 → 8; 0–100 → 80). The user is warned that observed score values may not reveal the questionnaire's full scale.
- **Exploratory clusters:** chooses k by the best silhouette score across candidate KMeans values when data quality, sample size and feature count support it; otherwise displays a disclosed, editable fallback.

Settings remain editable. Their widget keys include the uploaded file hash, worksheet, and detected mappings, so changing uploads/worksheets refreshes the suggestions instead of retaining settings from an earlier file. The UI displays the method and assumptions next to each control.

Validation after this change: **80 tests passed**; `app.py` and the new helper module pass Python syntax compilation. Live Streamlit browser interaction remains unverified in this build environment.
