from __future__ import annotations
import hashlib, io, json, os
from datetime import datetime
from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.express as px

from customer_analytics.full_analytics import (
    APP_TITLE, FIELD_DEFS, NOT_MAPPED, guess_column, read_uploaded_file,
    quality_report, apply_cleaning, feasibility, compute_metrics, make_report_text, outlier_record_report,
)
from customer_analytics.automatic_interpretation import interpret_all, render_interpretation_markdown, render_html_report
from customer_analytics.crm_intelligence import (
    FEATURES, load_knowledge_base, validate_knowledge_base, search_capabilities,
    compare_capabilities, assess_gap,
)
from customer_analytics.llm_provider import is_configured, narrate_computed_results
from customer_analytics.statistical_analysis import run_statistical_test, StatisticalAnalysisError
from customer_analytics.predictive_modeling import train_churn_models, ChurnModelError
from customer_analytics.exporting import excel_bundle
from customer_analytics.visualizations import build_analysis_chart
from customer_analytics.auto_defaults import recommend_analysis_defaults

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
KB_PATH = DATA_DIR / "crm_capabilities.json"
SAMPLE_PATH = DATA_DIR / "synthetic_customer_transactions.csv"

st.set_page_config(page_title="Customer Analytics & CRM Intelligence Agent", page_icon="◎", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
.block-container{max-width:1480px;padding-top:1.6rem;padding-bottom:3rem}
:root{--ink:#202124;--muted:#6b7280;--line:#e5e7eb;--surface:#f8f9fa}
h1,h2,h3{letter-spacing:-.025em} [data-testid="stMetric"]{border:1px solid var(--border-color, #e5e7eb);padding:14px 16px;border-radius:12px;background:var(--secondary-background-color, #f8f9fa);color:var(--text-color, #202124)} [data-testid="stMetric"] *{color:var(--text-color, #202124)}
[data-testid="stSidebar"]{border-right:1px solid #e5e7eb}
div[data-testid="stExpander"]{border:1px solid #e5e7eb;border-radius:10px}
.small-note{font-size:.88rem;color:#6b7280}
</style>
""", unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def get_kb():
    return load_knowledge_base(KB_PATH)

@st.cache_data(show_spinner=False)
def get_sample():
    return pd.read_csv(SAMPLE_PATH)

def csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")

def status_label(status: str) -> str:
    return {
        "verified_available":"Verified available",
        "available_plan_restrictions":"Available with plan restrictions",
        "available_addon":"Available through add-on",
        "available_integration":"Available through integration",
        "requires_configuration_or_custom_development":"Requires configuration/custom development",
        "not_verified":"Not verified — not proof of absence",
        "verified_unavailable_within_scope":"Verified unavailable within researched scope",
    }.get(status,status)

def safe_mapping(df: pd.DataFrame, prefix: str) -> dict[str,str]:
    mapping={}
    options=[NOT_MAPPED]+list(df.columns)
    for key,(label,aliases) in FIELD_DEFS.items():
        suggested=guess_column(df,aliases)
        default=suggested if suggested in options else NOT_MAPPED
        idx=options.index(default)
        mapping[key]=st.selectbox(label,options,index=idx,key=f"{prefix}_{key}")
    return mapping

def markdown_report_text(filename, original, cleaned, missing, outliers, audit, feas_df, results, interpretations):
    base=make_report_text(filename, original, cleaned, missing, outliers, audit, feas_df, results)
    return base+"\n\n"+render_interpretation_markdown(interpretations)

kb=get_kb()
kb_errors=validate_knowledge_base(kb)
verified_n=sum(x.get("verification_status")=="verified" for x in kb.get("capabilities",[]))
all_n=len(kb.get("capabilities",[]))
platform_names=[x["platform"] for x in kb.get("platforms",[])]
feature_lookup={x[0]:x for x in FEATURES}

with st.sidebar:
    st.markdown("# CRM Intelligence")
    st.caption("Customer Analytics & CRM Intelligence Agent")
    page=st.radio("Navigate",[
        "Overview","CRM Intelligence","Capability Comparison","Gap Assessment",
        "Customer Analytics","Data Quality","AI Insights & Recommendations","Reports & Exports","Settings & Sources"
    ],label_visibility="collapsed")
    st.divider()
    st.caption("Evidence-first prototype")
    st.caption(f"Knowledge base: {verified_n}/{all_n} feature claims verified")

if page == "Overview":
    st.title("Customer Analytics & CRM Intelligence Agent")
    st.write("A practical decision-support workspace that combines verified CRM capability evidence with customer-data analytics, transparent feasibility checks and automatically generated interpretations.")
    st.markdown("---")
    c1,c2,c3,c4=st.columns(4)
    c1.metric("CRM platforms",len(platform_names))
    c2.metric("Capability comparisons",all_n)
    c3.metric("Feature claims verified",verified_n)
    c4.metric("Analytics families",17)
    st.subheader("Workspace overview")
    left,right=st.columns([1.2,1])
    with left:
        st.markdown("**Start with a business question**")
        st.write("Describe a retention, conversion, engagement or service problem. The agent maps it to documented CRM capabilities and distinguishes evidence gaps from confirmed configuration or data problems.")
        st.write("**Then add customer data**")
        st.write("Upload CSV or Excel data, verify the suggested columns, inspect the quality report, run supported analyses and receive findings and recommended next steps automatically.")
    with right:
        st.markdown("**Current research coverage**")
        st.write(f"The current curated knowledge base contains {all_n} platform-capability pairs across {len(platform_names)} CRM platforms. {verified_n} claims have a feature-specific official source; the rest are explicitly marked **not verified**.")
        if kb_errors: st.error(f"Knowledge-base validation: {len(kb_errors)} error(s). See Settings & Sources.")
        else: st.success("Knowledge-base schema validation passed.")
    st.info("This prototype complements CRM platforms. It does not connect to or modify a live CRM account. Synthetic demonstration data are labelled and remain separate from real business evidence.")
    st.subheader("Demonstration data")
    demo=get_sample()
    st.caption("Synthetic demonstration dataset — no real customer records.")
    st.dataframe(demo.head(8),use_container_width=True,hide_index=True)
    st.download_button("Download synthetic customer dataset",csv_bytes(demo),file_name="synthetic_customer_transactions.csv",mime="text/csv")

elif page == "CRM Intelligence":
    st.title("CRM Intelligence")
    st.write("Search the current evidence base by business objective, workflow, or capability. Search results preserve the source status and distinguish confirmed product evidence from research work still outstanding.")
    q=st.text_input("What capability or business problem are you investigating?",placeholder="e.g. identify customers at risk of churn and assign retention follow-ups")
    selected_platforms=st.multiselect("CRM platforms",platform_names,default=platform_names)
    if q.strip():
        found=search_capabilities(q,selected_platforms,data=kb)
        if not found: st.warning("No matching records found. Try broader wording or inspect the full comparison matrix.")
        else:
            display=[]
            for r in found[:50]:
                display.append({"Platform":r["platform"],"Capability":r["capability_name"],"Status":status_label(r["availability_status"]),"Evidence":r["verification_status"],"Plan scope":r.get("plan_scope",""),"Evidence note":r.get("evidence_note",""),"Source":r.get("evidence_url","")})
            st.dataframe(pd.DataFrame(display),use_container_width=True,hide_index=True,column_config={"Source":st.column_config.LinkColumn("Source",display_text="Open source")})
            st.download_button("Download search results",csv_bytes(pd.DataFrame(display)),file_name="crm_intelligence_search.csv",mime="text/csv")
    else:
        st.info("Enter a business question to search. Examples: churn prediction, workflow automation, dashboards, customer health score, email journeys, data quality.")
    with st.expander("Known evidence examples",expanded=False):
        verified=[r for r in kb["capabilities"] if r["verification_status"]=="verified"]
        st.dataframe(pd.DataFrame([{"Platform":r["platform"],"Capability":r["capability_name"],"Status":status_label(r["availability_status"]),"Scope":r["plan_scope"],"Verified date":r["verification_date"],"Source":r["evidence_url"]} for r in verified]),use_container_width=True,hide_index=True,column_config={"Source":st.column_config.LinkColumn("Source",display_text="Open official documentation")})

elif page == "Capability Comparison":
    st.title("Capability Comparison")
    st.write("Select required capabilities and compare the evidence recorded for each CRM. An unresearched feature stays **Not verified**; it is never treated as unavailable by default.")
    platforms=st.multiselect("Select platforms to compare",platform_names,default=platform_names)
    options=[f[0] for f in FEATURES]
    defaults=[x[0] for x in FEATURES]
    feature_ids=st.multiselect("Business capabilities",options,default=defaults,format_func=lambda value:feature_lookup[value][1])
    if platforms and feature_ids:
        comparison=compare_capabilities(platforms,feature_ids,data=kb)
        view=pd.DataFrame([{"Capability":r["capability"],"Category":r["category"],"Platform":r["platform"],"Status":status_label(r["status"]),"Evidence status":r["verification_status"],"Plan / scope":r["plan_scope"],"Implementation type":r["implementation_type"],"Limitations":r["limitations"],"Source URL":r["evidence_url"],"Source note":r["evidence_note"],"Verified on":r["verification_date"]} for r in comparison])
        st.dataframe(view,use_container_width=True,hide_index=True,column_config={"Source URL":st.column_config.LinkColumn("Source URL",display_text="Open source")})
        st.caption(f"{sum(r['verification_status']=='verified' for r in comparison)} of {len(comparison)} selected comparisons have feature-specific verified evidence. Coverage varies by product and capability.")
        st.download_button("Download capability comparison",csv_bytes(view),file_name="crm_capability_comparison.csv",mime="text/csv")
        buf=io.BytesIO()
        with pd.ExcelWriter(buf,engine="openpyxl") as wr: view.to_excel(wr,index=False,sheet_name="Comparison")
        st.download_button("Download Excel comparison",buf.getvalue(),file_name="crm_capability_comparison.xlsx",mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

elif page == "Gap Assessment":
    st.title("Gap Assessment")
    st.write("Use documented feature evidence plus what your team knows about its current plan, configuration, data and workflow. Prioritization is a transparent stakeholder-scored heuristic, not a measured financial estimate.")
    with st.form("gap_form"):
        p=st.selectbox("Current CRM platform",platform_names)
        feat=st.selectbox("Closest capability to the business requirement",options,format_func=lambda value:feature_lookup[value][1])
        req=st.text_area("Business requirement / desired outcome",placeholder="We want to identify at-risk customers and give the retention team a prioritized queue.",height=90)
        config_label=st.selectbox("What is known about the current setup?",[
            "Unknown / not assessed","Documented capability is configured and used","Capability appears available but is not configured","Plan or add-on entitlement is unknown","Team reports it is unavailable in the current plan","Configured but the business outcome remains inadequate"
        ])
        current_config={"Unknown / not assessed":"unknown","Documented capability is configured and used":"configured","Capability appears available but is not configured":"not_configured","Plan or add-on entitlement is unknown":"plan_unknown","Team reports it is unavailable in the current plan":"unavailable_reported","Configured but the business outcome remains inadequate":"configured_ineffective"}[config_label]
        a,b,c=st.columns(3)
        impact=a.slider("Business impact",1,5,4)
        frequency=b.slider("Problem frequency",1,5,3)
        manual_effort=c.slider("Manual effort",1,5,3)
        d,e,f=st.columns(3)
        consequences=d.slider("Customer / financial consequence",1,5,4)
        differentiation=e.slider("Differentiation from existing capability",1,5,3)
        feasibility_rating=f.slider("Implementation feasibility",1,5,3)
        ga,gb=st.columns(2)
        analytics_relevance=ga.slider("Customer analytics relevance",1,5,5)
        data_ready=gb.selectbox("Required data available?",["Unknown","Yes","No"],index=0)
        with st.expander("Adjust prioritization weights (default totals 100%)", expanded=False):
            st.caption("Weights change the stakeholder-priority score only; they are not statistical confidence or predicted ROI. At least one weight must be non-zero.")
            wdefaults={"impact":30,"frequency":15,"manual_effort":15,"customer_financial_consequence":15,"differentiation":10,"feasibility":10,"customer_analytics_relevance":5}
            weight_cols=st.columns(4)
            weight_labels=[("impact","Business impact"),("frequency","Frequency"),("manual_effort","Manual effort"),("customer_financial_consequence","Customer/financial consequence"),("differentiation","Differentiation"),("feasibility","Feasibility"),("customer_analytics_relevance","Analytics relevance")]
            weights={}
            for wi,(wk,wl) in enumerate(weight_labels):
                with weight_cols[wi%4]:
                    weights[wk]=st.number_input(wl,min_value=0,max_value=100,value=wdefaults[wk],step=5,key=f"gap_weight_{wk}")
            st.caption(f"Current total: {sum(weights.values())}% (weights are normalized automatically)")
        integration=st.checkbox("A third-party system/data integration appears necessary",value=False)
        process=st.checkbox("Workflow, ownership or adoption may be part of the problem",value=False)
        submit=st.form_submit_button("Assess and prioritize gap")
    if submit:
        record=next((r for r in kb["capabilities"] if r["platform"]==p and r["feature_id"]==feat),None)
        payload={"evidence_verified":bool(record and record["verification_status"]=="verified"),
                 "availability_status":record["availability_status"] if record else "not_verified",
                 "configuration_status":current_config,
                 "data_available":False if data_ready=="No" else (True if data_ready=="Yes" else None),
                 "integration_needed":integration,"process_issue":process,
                 "impact":impact,"frequency":frequency,"manual_effort":manual_effort,
                 "customer_financial_consequence":consequences,"differentiation":differentiation,
                 "feasibility":feasibility_rating,"customer_analytics_relevance":analytics_relevance,
                 "weights":{k:v/100 for k,v in weights.items()},
                 "evidence_confidence":record.get("evidence_confidence",1) if record else 1}
        try:
            result=assess_gap(payload)
            st.session_state["latest_gap_assessment"]={"platform":p,"feature":feature_lookup[feat][1],"requirement":req,"record":record,"result":result,"ratings":payload}
        except ValueError as exc:
            st.error(f"Prioritization could not be completed: {exc}")
    item=st.session_state.get("latest_gap_assessment")
    if item:
        result=item["result"]
        st.subheader("Assessment result")
        x1,x2,x3=st.columns(3); x1.metric("Gap class",result["gap_type"]); x2.metric("Priority score",f"{result['priority_score']}/100"); x3.metric("Priority",result["priority"])
        st.markdown(f"**Requirement:** {item['requirement'] or 'Not provided'}")
        st.markdown(f"**Assessment:** {result['description']}")
        st.markdown(f"**Confirmed finding:** {'Yes, within the provided evidence scope' if result['confirmed_finding'] else 'No — hypothesis or verification task'}")
        st.markdown(f"**Evidence confidence:** {result['evidence_confidence']}")
        st.markdown(f"**Recommended next step:** {result['recommended_next_step']}")
        record=item.get("record")
        if record:
            st.markdown(f"**Evidence record:** {status_label(record['availability_status'])} · {record.get('plan_scope','')}")
            st.write(record.get("evidence_note",""))
            if record.get("evidence_url"): st.markdown(f"[Open official evidence]({record['evidence_url']})")
        st.warning("Priority score reflects adjustable 1–5 stakeholder ratings and user-selected weights. It is not an ROI estimate, statistical confidence score or proof of a capability gap.")
        report={"assessment_date":datetime.now().isoformat(timespec="seconds"),**item}
        st.download_button("Download gap assessment JSON",json.dumps(report,indent=2,default=str).encode(),file_name="crm_gap_assessment.json",mime="application/json")

elif page == "Customer Analytics":
    st.title("Customer Analytics Workspace")
    st.write("Upload data, verify suggested column mappings, review quality issues, and run the available analyses. Metrics are calculated in Python; each completed result receives an automatic interpretation.")
    uploaded=st.file_uploader("Upload customer data (CSV, XLSX, XLSM, XLS)",type=["csv","xlsx","xlsm","xls"],key="customer_data_upload")
    if uploaded is not None:
        if len(uploaded.getvalue()) > 100*1024*1024:
            st.session_state.pop("uploaded_bytes", None); st.session_state.pop("uploaded_name", None); st.session_state.pop("upload_hash", None)
            st.error("File exceeds the prototype's 100 MB upload limit. Reduce or split the file before uploading.")
        else:
            current_hash=hashlib.sha256(uploaded.getvalue()).hexdigest()
            st.session_state["uploaded_bytes"]=uploaded.getvalue(); st.session_state["uploaded_name"]=uploaded.name; st.session_state["upload_hash"]=current_hash
    raw_bytes=st.session_state.get("uploaded_bytes")
    filename=st.session_state.get("uploaded_name","uploaded_dataset.csv")
    if raw_bytes is None:
        st.info("Upload a customer dataset above, or use the synthetic sample in Overview. The raw upload is held in this app session; no external AI call is made by default.")
    else:
        suffix=Path(filename).suffix.lower()
        sheets=[]
        if suffix in {".xlsx",".xlsm",".xls"}:
            try:
                sheets=pd.ExcelFile(io.BytesIO(raw_bytes)).sheet_names
            except Exception as exc:
                st.error(f"Excel workbook could not be opened: {exc}"); sheets=[]
        selected_sheet=None
        if sheets:
            selected_sheet=st.selectbox("Worksheet",sheets,key=f"sheet_{st.session_state['upload_hash']}")
        class UploadedAdapter:
            def __init__(self,name,data): self.name=name; self._data=data
            def getvalue(self): return self._data
        adapter=UploadedAdapter(filename,raw_bytes)
        try:
            raw_df,_=read_uploaded_file(adapter,selected_sheet)
        except Exception as exc:
            st.error(f"Could not read upload: {exc}"); raw_df=pd.DataFrame()
        if raw_df.empty or raw_df.shape[1]==0:
            st.warning("The selected worksheet has no usable rows/columns. Choose another worksheet or inspect the source file.")
        else:
            st.session_state["active_dataset"]={"name":filename,"sheet":selected_sheet or "CSV","raw":raw_df.copy(deep=True)}
            st.markdown(f"**{filename}** · {raw_df.shape[0]:,} rows × {raw_df.shape[1]:,} columns")
            st.dataframe(raw_df.head(15),use_container_width=True,hide_index=True)
            # Auto-suggest settings immediately from the current upload. The column
            # mappings below may be overridden by the user, so these notes explicitly
            # disclose which suggestions are based on inferred mappings.
            inferred_mapping={key:guess_column(raw_df,aliases) for key,(label,aliases) in FIELD_DEFS.items()}
            auto_defaults=recommend_analysis_defaults(raw_df,inferred_mapping)
            auto_signature=hashlib.sha256((st.session_state["upload_hash"]+"|"+str(selected_sheet or "CSV")+"|"+json.dumps(inferred_mapping,sort_keys=True,default=str)).encode("utf-8")).hexdigest()[:12]
            st.caption("The three analysis settings below are auto-suggested from this file. They remain editable; the notes disclose the evidence and any fallback assumptions.")
            with st.form(f"analytics_form_{st.session_state['upload_hash']}_{selected_sheet or 'csv'}"):
                st.subheader("Confirm semantic column mappings")
                st.caption("Suggestions are based on column-name aliases only. Confirm them; ambiguous business definitions are not inferred automatically.")
                map_cols=st.columns(3)
                mapping={}
                for i,(key,(label,aliases)) in enumerate(FIELD_DEFS.items()):
                    with map_cols[i%3]:
                        suggested=guess_column(raw_df,aliases)
                        col_options=[NOT_MAPPED]+list(raw_df.columns)
                        default=suggested if suggested in col_options else NOT_MAPPED
                        mapping[key]=st.selectbox(label,col_options,index=col_options.index(default),key=f"{st.session_state['upload_hash']}_{selected_sheet}_{key}")
                st.subheader("Analysis options")
                o1,o2,o3=st.columns(3)
                inactivity_days=o1.number_input("Inactivity proxy window (days) — auto-suggested",min_value=1,max_value=1000,value=int(auto_defaults["inactivity_days"]),key=f"inactivity_days_{auto_signature}")
                csat_threshold=o2.number_input("Satisfied CSAT threshold — auto-suggested",min_value=0.0,max_value=100.0,value=float(auto_defaults["csat_threshold"]),step=0.5,key=f"csat_threshold_{auto_signature}")
                clusters=o3.number_input("Exploratory clusters — auto-suggested",min_value=2,max_value=8,value=int(auto_defaults["n_clusters"]),key=f"cluster_count_{auto_signature}")
                with o1:
                    st.caption(auto_defaults["inactivity_note"])
                with o2:
                    st.caption(auto_defaults["csat_note"])
                with o3:
                    st.caption(auto_defaults["clusters_note"])
                row_granularity=st.selectbox("What does one row represent?",["One transaction / order per row","One order line / item per row","One customer per row","One event or support ticket per row","Mixed / unsure"],index=0,help="This definition affects AOV and purchase-frequency metrics. If unsure, choose Mixed / unsure so the application avoids unsupported aggregation.")
                with st.expander("Optional CAC inputs (only if costs and acquired customers are aligned)",expanded=False):
                    s1,s2=st.columns(2)
                    manual_spend_input=s1.number_input("Verified aggregate acquisition spend",min_value=0.0,value=0.0,step=100.0,help="Leave at 0 unless this is a verified total for a defined period/channel/currency.")
                    manual_new_input=s2.number_input("Verified new customers acquired",min_value=0,value=0,step=1,help="Must cover the same period and attribution scope as the spend total.")
                    cac_inputs_confirmed=st.checkbox("I confirm mapped spend/count columns are additive and aligned, if using them",value=False)
                remove_dupes=st.checkbox("Remove exact duplicate rows from analysis copy (original preserved)",value=False)
                outlier_method=st.selectbox("Optional outlier treatment",["Flag only (recommended)","Cap to IQR bounds","Replace flagged values with column median","Drop rows with any eligible IQR outlier"],index=0)
                st.caption("Non-default outlier treatments can distort valid high-value customers. Compare results against the original and document the reason before using them.")
                run=st.form_submit_button("Profile, assess feasibility, analyse and interpret")
            if run:
                # Never overwrite or mutate the original upload.
                original=raw_df.copy(deep=True)
                missing,outliers=quality_report(original)
                cleaned,audit=apply_cleaning(original,remove_dupes,outlier_method)
                outlier_records=outlier_record_report(original,outlier_method)
                manual_spend = float(manual_spend_input) if manual_spend_input > 0 else None
                manual_new_customers = int(manual_new_input) if manual_new_input > 0 else None
                extra={"manual_spend":manual_spend,"manual_new_customers":manual_new_customers,"row_granularity":row_granularity}
                feasibility_df=feasibility(mapping,cleaned,{**extra,"inactivity_days":int(inactivity_days)})
                options={"inactivity_days":int(inactivity_days),"csat_threshold":float(csat_threshold),"n_clusters":int(clusters),
                         "row_granularity":row_granularity,"manual_spend":manual_spend,"manual_new_customers":manual_new_customers,
                         "cac_inputs_confirmed":bool(cac_inputs_confirmed)}
                results,meta=compute_metrics(cleaned,mapping,options)
                interpretations=interpret_all(results,dataset_rows=len(cleaned))
                # If a metric has no output, feasibility remains useful and no result is fabricated.
                report_text=markdown_report_text(filename,original,cleaned,missing,outliers,audit,feasibility_df,results,interpretations)
                report_html=render_html_report("Customer Analytics & CRM Intelligence Report",report_text,render_interpretation_markdown(interpretations))
                duplicate_mask=original.duplicated(keep=False)
                duplicate_rows=original.loc[duplicate_mask].copy()
                if not duplicate_rows.empty:
                    # Original row number includes the header row as row 1, for CSV/standard Excel exports.
                    duplicate_rows.insert(0,"Source_Row_Number",(duplicate_rows.index.to_series()+2).astype(int))
                    first_duplicate=original.duplicated(keep="first")
                    duplicate_rows["Review_Decision"]= [
                        ("Removed from analysis copy" if bool(first_duplicate.loc[idx]) else "Retained as first occurrence") if remove_dupes else "Retained pending human review"
                        for idx in duplicate_rows.index]
                    duplicate_rows["Reason"]="Exact full-row duplicate candidate; identical rows may still require business-owner review."
                st.session_state["analysis_bundle"]={"filename":filename,"sheet":selected_sheet,"original":original,"cleaned":cleaned,"missing":missing,"outliers":outliers,"outlier_records":outlier_records,"audit":audit,"feasibility":feasibility_df,"results":results,"interpretations":interpretations,"report_text":report_text,"report_html":report_html,"metadata":{**meta,"mapping":mapping,"options":options,"generated_at":datetime.now().isoformat(timespec="seconds"),"source_file_hash":st.session_state['upload_hash']},"duplicate_rows":duplicate_rows}
                st.success(f"Analysis completed: {len(results)} output(s) calculated with automatic interpretations. Open AI Insights & Recommendations for the consolidated findings.")
            active=st.session_state.get("analysis_bundle")
            if active and active.get("filename")==filename and active.get("sheet")== (selected_sheet or "CSV"):
                st.subheader("Calculated analyses")
                if not active["results"]: st.warning("No supported metric produced a result for the current mapping. Review feasibility and mapping; no unsupported metric was fabricated.")
                for name,r in active["results"].items():
                    with st.expander(f"{name} — computed result",expanded=False):
                        st.write(r.get("summary",""))
                        if r.get("note"): st.caption("Method / limitation: "+r["note"])
                        table=r.get("table")
                        chart=build_analysis_chart(name,r)
                        if chart is not None: st.plotly_chart(chart,use_container_width=True)
                        if isinstance(table,pd.DataFrame) and not table.empty: st.dataframe(table.head(500),use_container_width=True,hide_index=True)
                        st.markdown("**Automatic interpretation**")
                        interp=active["interpretations"].get(name,{})
                        st.write(interp.get("business_implications",""))
                        for recommendation in interp.get("recommendations",[]): st.markdown(f"- {recommendation}")
                st.subheader("Analysis feasibility")
                st.dataframe(active["feasibility"],use_container_width=True,hide_index=True)
            active=st.session_state.get("analysis_bundle")
            if active and active.get("filename")==filename and active.get("sheet")== (selected_sheet or "CSV"):
                with st.expander("Optional statistical testing",expanded=False):
                    st.write("Run one explicit inferential analysis on selected columns. The app does not test every possible pair automatically.")
                    numeric_cols=[c for c in active["cleaned"].columns if pd.to_numeric(active["cleaned"][c],errors="coerce").notna().sum()>=3 and pd.to_numeric(active["cleaned"][c],errors="coerce").nunique()>=2]
                    categorical_cols=[c for c in active["cleaned"].columns if 2<=active["cleaned"][c].nunique(dropna=True)<=20 and (not pd.api.types.is_numeric_dtype(active["cleaned"][c]))]
                    method=st.selectbox("Test",["Pearson correlation","Linear regression (one predictor)","Chi-square independence test"],key="stat_method")
                    stat_x=stat_y=cat_x=cat_y=None
                    if method in {"Pearson correlation","Linear regression (one predictor)"}:
                        if len(numeric_cols)>=2:
                            sx,sy=st.columns(2)
                            stat_x=sx.selectbox("X / predictor",numeric_cols,index=0,key="stat_x")
                            stat_y=sy.selectbox("Y / outcome",numeric_cols,index=1,key="stat_y")
                        else: st.info("At least two numeric columns with nonconstant values are required.")
                    else:
                        if len(categorical_cols)>=2:
                            cx,cy=st.columns(2)
                            cat_x=cx.selectbox("First category",categorical_cols,index=0,key="stat_cat_x")
                            cat_y=cy.selectbox("Second category",categorical_cols,index=1,key="stat_cat_y")
                        else: st.info("At least two categorical columns with 2–20 categories each are required. Do not choose customer or transaction IDs.")
                    if st.button("Run selected statistical test",key="run_statistical_test"):
                        try:
                            stat_result=run_statistical_test(active["cleaned"],method,x=stat_x,y=stat_y,category_x=cat_x,category_y=cat_y)
                            test_name=f"Statistical test — {method}"
                            active["results"][test_name]=stat_result
                            active["interpretations"][test_name]=interpret_all({test_name:stat_result},dataset_rows=len(active["cleaned"]))[test_name]
                            active["metadata"]["last_statistical_test"]={"method":method,"completed_at":datetime.now().isoformat(timespec="seconds"),"output_summary":stat_result["summary"]}
                            active["report_text"]=markdown_report_text(active["filename"],active["original"],active["cleaned"],active["missing"],active["outliers"],active["audit"],active["feasibility"],active["results"],active["interpretations"])
                            active["report_html"]=render_html_report("Customer Analytics & CRM Intelligence Report",active["report_text"],render_interpretation_markdown(active["interpretations"]))
                            st.session_state["analysis_bundle"]=active
                            st.success("Statistical test completed. Its computed output and interpretation have been added to the report.")
                            st.write(stat_result["summary"])
                            if stat_result.get("note"): st.caption(stat_result["note"])
                            stat_chart=build_analysis_chart(test_name,stat_result)
                            if stat_chart is not None: st.plotly_chart(stat_chart,use_container_width=True)
                            if isinstance(stat_result.get("table"),pd.DataFrame): st.dataframe(stat_result["table"],use_container_width=True,hide_index=True)
                            for action in active["interpretations"][test_name].get("recommendations",[]): st.markdown(f"- {action}")
                        except (StatisticalAnalysisError,ValueError,TypeError) as exc:
                            st.error(f"Statistical test was not completed: {exc}")

                # Predictive churn modelling is offered only when an explicit binary outcome is mapped.
                active=st.session_state.get("analysis_bundle")
                if active and active.get("filename")==filename and active.get("sheet")== (selected_sheet or "CSV"):
                    model_mapping=active["metadata"].get("mapping",{})
                    target_col=model_mapping.get("churn_flag")
                    customer_col=model_mapping.get("customer_id")
                    model_data=active["cleaned"]
                    with st.expander("Optional predictive churn model (explicit label required)",expanded=False):
                        st.write("Compares logistic regression with random forest on a stratified holdout set. It does not create a churn label from inactivity. Select only predictors measured before the outcome window.")
                        if target_col in model_data.columns and customer_col in model_data.columns:
                            target_values=sorted(model_data[target_col].dropna().astype(str).str.strip().unique().tolist())
                            # Keep the default feature shortlist conservative and exclude obvious time/ID fields.
                            feature_candidates=[]
                            for col_name in model_data.columns:
                                if col_name in {target_col,customer_col}: continue
                                low_name=str(col_name).casefold()
                                if any(term in low_name for term in ["date","time","timestamp","created","resolved","email","phone","address","_id"," id"]): continue
                                ser=model_data[col_name]
                                if pd.api.types.is_numeric_dtype(ser):
                                    if ser.nunique(dropna=True)>=2: feature_candidates.append(col_name)
                                elif 2<=ser.nunique(dropna=True)<=25:
                                    feature_candidates.append(col_name)
                            if len(target_values)==2 and feature_candidates:
                                with st.form("predictive_churn_form"):
                                    positive_label=st.selectbox("Which target value means churn?",target_values,key="churn_positive_class")
                                    selected_features=st.multiselect("Predictor fields (do not select identifiers or post-churn outcomes)",feature_candidates,default=feature_candidates[:min(5,len(feature_candidates))],key="churn_model_features")
                                    feature_timing_confirmed=st.checkbox("I confirm these predictors are available before the churn outcome / prediction window.",value=False)
                                    model_submit=st.form_submit_button("Train and evaluate exploratory churn models")
                                if model_submit:
                                    try:
                                        model_result=train_churn_models(model_data,customer_id_col=customer_col,target_col=target_col,positive_label=positive_label,feature_cols=selected_features,feature_timing_confirmed=feature_timing_confirmed)
                                        model_name="Predictive Churn Model Evaluation"
                                        interp=interpret_all({model_name:{"summary":model_result["summary"],"table":model_result["metrics"],"note":model_result["note"]}},dataset_rows=len(model_result["customer_level_dataset"]))[model_name]
                                        active["results"][model_name]={"summary":model_result["summary"],"table":model_result["metrics"],"note":model_result["note"],"auxiliary_tables":{
                                            "Feature Importance":model_result["feature_importance"],"Holdout Predictions":model_result["predictions"],**{f"Confusion {k}":v for k,v in model_result["confusion_matrices"].items()}}}
                                        active["interpretations"][model_name]=interp
                                        active["metadata"]["predictive_churn_model"]={"positive_label":model_result["positive_label"],"negative_label":model_result["negative_label"],"features":model_result["features"],"sample_size":model_result["sample_size"],"validation":"stratified random holdout; use time-based validation before deployment"}
                                        active["report_text"]=markdown_report_text(active["filename"],active["original"],active["cleaned"],active["missing"],active["outliers"],active["audit"],active["feasibility"],active["results"],active["interpretations"])
                                        active["report_html"]=render_html_report("Customer Analytics & CRM Intelligence Report",active["report_text"],render_interpretation_markdown(active["interpretations"]))
                                        st.session_state["analysis_bundle"]=active
                                        st.success("Models evaluated on a holdout set. Review all metrics and limitations before use; results are not production-ready churn probabilities.")
                                    except (ChurnModelError,ValueError,TypeError) as exc:
                                        st.error(f"Churn modelling was not completed: {exc}")
                                current_model=st.session_state.get("analysis_bundle",{}).get("results",{}).get("Predictive Churn Model Evaluation")
                                if current_model:
                                    st.markdown("**Holdout model comparison**")
                                    model_chart=build_analysis_chart("Predictive Churn Model Evaluation",current_model)
                                    if model_chart is not None: st.plotly_chart(model_chart,use_container_width=True)
                                    st.dataframe(current_model["table"],use_container_width=True,hide_index=True)
                                    st.markdown("**Automatic interpretation**")
                                    st.write(st.session_state["analysis_bundle"]["interpretations"]["Predictive Churn Model Evaluation"].get("business_implications",""))
                                    for action in st.session_state["analysis_bundle"]["interpretations"]["Predictive Churn Model Evaluation"].get("recommendations",[]): st.markdown(f"- {action}")
                                    for aux_name,aux_df in current_model.get("auxiliary_tables",{}).items():
                                        with st.expander(aux_name,expanded=False): st.dataframe(aux_df,use_container_width=True,hide_index=True)
                            else:
                                st.info("To run this model, map a stable customer ID and an explicit binary churn/cancellation outcome, provide at least two outcome classes, and select valid predictors. An inactivity proxy alone is not a supervised target.")
                        else:
                            st.info("A mapped customer ID and explicit churn/cancellation target are required. Current inactivity-rule results remain descriptive proxies and cannot replace a verified target.")

elif page == "Data Quality":
    st.title("Data Quality & Audit")
    bundle=st.session_state.get("analysis_bundle")
    dataset=st.session_state.get("active_dataset")
    if bundle:
        st.write(f"Latest analysed dataset: **{bundle['filename']}** ({bundle['sheet']})")
        a,b,c,d=st.columns(4)
        a.metric("Original rows",f"{len(bundle['original']):,}")
        b.metric("Analysis rows",f"{len(bundle['cleaned']):,}")
        c.metric("Exact duplicate rows",f"{int(bundle['original'].duplicated().sum()):,}")
        d.metric("Columns with missing values",f"{int((bundle['missing']['missing_count']>0).sum()):,}")
        st.subheader("Missingness and schema")
        st.dataframe(bundle["missing"],use_container_width=True,hide_index=True)
        st.subheader("Potential numeric outliers (flags are not proof of error)")
        st.dataframe(bundle["outliers"],use_container_width=True,hide_index=True)
        st.subheader("Row-level outlier review queue")
        st.caption("Each flagged numeric cell records source row, bounds, selected treatment, and whether the treatment was applied to the analysis copy.")
        outlier_records=bundle.get("outlier_records",pd.DataFrame())
        if outlier_records.empty: st.success("No eligible numeric outlier candidates were flagged under the configured 1.5×IQR rule.")
        else:
            st.dataframe(outlier_records.head(500),use_container_width=True,hide_index=True)
            st.download_button("Download row-level outlier audit",csv_bytes(outlier_records),file_name="outlier_record_audit.csv",mime="text/csv")
        st.subheader("Exact duplicate rows")
        st.caption("Repeated customer IDs or orders are not automatically duplicates; only exact full-row duplicates are flagged here.")
        if bundle["duplicate_rows"].empty: st.success("No exact duplicate rows detected.")
        else:
            st.dataframe(bundle["duplicate_rows"].head(300),use_container_width=True,hide_index=True)
            st.download_button("Download exact duplicate row report",csv_bytes(bundle["duplicate_rows"]),file_name="duplicate_row_audit.csv",mime="text/csv")
        st.subheader("Cleaning audit")
        for item in bundle["audit"]: st.markdown(f"- {item}")
        st.download_button("Download missingness report",csv_bytes(bundle["missing"]),file_name="missingness_report.csv",mime="text/csv")
        st.download_button("Download outlier report",csv_bytes(bundle["outliers"]),file_name="outlier_flags.csv",mime="text/csv")
        if not bundle.get("outlier_records",pd.DataFrame()).empty:
            st.download_button("Download row-level outlier decisions",csv_bytes(bundle["outlier_records"]),file_name="outlier_record_audit.csv",mime="text/csv")
        st.download_button("Download cleaned analysis copy",csv_bytes(bundle["cleaned"]),file_name="cleaned_analysis_copy.csv",mime="text/csv")
    elif dataset:
        df=dataset["raw"]; missing,outliers=quality_report(df)
        st.write(f"{dataset['name']} · {len(df):,} rows × {df.shape[1]:,} columns")
        st.dataframe(missing,use_container_width=True,hide_index=True)
        st.dataframe(outliers,use_container_width=True,hide_index=True)
    else: st.info("Upload a file and run the analysis workflow to generate a complete quality audit. The original file is kept unchanged in session memory.")

elif page == "AI Insights & Recommendations":
    st.title("AI Insights & Recommendations")
    bundle=st.session_state.get("analysis_bundle")
    if not bundle:
        st.info("Run an analysis first. Once metrics finish, this page automatically shows findings, implications, recommendations, KPIs and limitations.")
    else:
        st.caption(f"Interpretation based on actual computed results for {bundle['filename']} · {bundle['metadata']['generated_at']}")
        st.info("The default narrative is deterministic and derived from computed result summaries/tables. Recommendations are proposals, not claimed business outcomes.")
        for name,item in bundle["interpretations"].items():
            with st.expander(name,expanded=True):
                st.markdown("**Overview**"); st.write(item.get("overview",""))
                st.markdown("**Key findings**")
                for fact in item.get("key_findings",[]): st.markdown(f"- {fact}")
                st.markdown("**Business implications**"); st.write(item.get("business_implications",""))
                st.markdown("**Root-cause assessment**"); st.write(item.get("root_cause_assessment",""))
                st.markdown("**Recommended actions**")
                for action in item.get("recommendations",[]): st.markdown(f"- {action}")
                st.markdown("**KPIs to monitor**")
                for kpi in item.get("kpis",[]): st.markdown(f"- {kpi}")
                st.markdown("**Limitations and confidence notes**")
                for limitation in item.get("limitations",[]): st.markdown(f"- {limitation}")
        if is_configured():
            if st.button("Enhance wording with optional OpenAI model (aggregate findings only)"):
                try:
                    llm_text=narrate_computed_results(bundle["interpretations"])
                    st.session_state["llm_interpretations"]={"generated_at":datetime.now().isoformat(timespec="seconds"),"content":llm_text}
                    st.success("Optional AI narration returned. Deterministic calculations and the original interpretation were retained.")
                except Exception as exc: st.error(f"Optional AI narration unavailable: {exc}. Deterministic interpretations remain available.")
        else:
            st.caption("Optional OpenAI narration is not configured. No API request is made; all core analytics and interpretations work locally.")
        if st.session_state.get("llm_interpretations"):
            with st.expander("Optional AI wording output",expanded=False): st.json(st.session_state["llm_interpretations"]["content"])

elif page == "Reports & Exports":
    st.title("Reports & Exports")
    bundle=st.session_state.get("analysis_bundle")
    if not bundle:
        st.info("No analysis report is ready yet. Run a dataset analysis, or export a CRM capability comparison from that page.")
    else:
        st.write("Exports include methods, data-quality checks, feasibility, calculated results, automatic interpretation, recommendations and limitations.")
        st.download_button("Download consolidated Markdown / text report",bundle["report_text"].encode("utf-8"),file_name="customer_analytics_report.md",mime="text/markdown")
        st.download_button("Download HTML report",bundle["report_html"].encode("utf-8"),file_name="customer_analytics_report.html",mime="text/html")
        st.download_button("Download original data copy",csv_bytes(bundle["original"]),file_name="original_upload_copy.csv",mime="text/csv")
        st.download_button("Download cleaned analysis copy",csv_bytes(bundle["cleaned"]),file_name="cleaned_analysis_copy.csv",mime="text/csv")
        xbytes=excel_bundle(bundle["original"],bundle["cleaned"],bundle["missing"],bundle["outliers"],bundle["feasibility"],bundle["results"],bundle["metadata"],duplicate_rows=bundle.get("duplicate_rows"),outlier_records=bundle.get("outlier_records"),audit=bundle.get("audit"),interpretations=bundle.get("interpretations"))
        st.download_button("Download full Excel analysis workbook",xbytes,file_name="customer_analytics_report.xlsx",mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        st.download_button("Download interpretation JSON",json.dumps(bundle["interpretations"],indent=2,default=str).encode(),file_name="automatic_interpretations.json",mime="application/json")
        for name,res in bundle["results"].items():
            table=res.get("table")
            if isinstance(table,pd.DataFrame) and not table.empty:
                st.download_button(f"Download {name} result CSV",csv_bytes(table),file_name=f"{name.lower().replace(' ','_').replace('/','_')}.csv",mime="text/csv",key="dl_"+hashlib.sha1(name.encode()).hexdigest()[:12])

elif page == "Settings & Sources":
    st.title("Settings & Research Sources")
    st.write("This project runs locally by default. Optional LLM narration can be enabled through environment variables; it is never called without a user-triggered action.")
    st.subheader("Knowledge-base quality")
    if kb_errors: st.error("Validation problems: "+"; ".join(kb_errors))
    else: st.success("CRM knowledge-base schema validation passed.")
    st.write(f"Capability pairs: {all_n}. Feature-level official-source verified claims: {verified_n}. Not verified: {all_n-verified_n}.")
    st.caption("A feature-level verified statement records what the linked official source supports. A general landing page is not evidence that the feature exists or is absent.")
    st.subheader("Platforms and source links")
    for p in kb["platforms"]:
        with st.expander(p["platform"]):
            st.markdown(f"[Official product page]({p['official_url']})")
            st.write(p.get("research_note",""))
            rows=[r for r in kb["capabilities"] if r["platform"]==p["platform"]]
            v=[r for r in rows if r["verification_status"]=="verified"]
            st.caption(f"{len(v)} verified feature-specific record(s) and {len(rows)-len(v)} awaiting verification")
            for r in v:
                st.markdown(f"**{r['capability_name']}** — {status_label(r['availability_status'])} · verified {r['verification_date']}")
                st.write(r["evidence_note"])
                st.markdown(f"[Official documentation]({r['evidence_url']})")
    st.subheader("Configuration")
    st.code("OPENAI_API_KEY=your_key_here\nOPENAI_MODEL=gpt-4.1-mini",language="bash")
    st.write("Place values in a local .env file only if you load it explicitly, or set environment variables in your terminal. Never commit actual secrets. Optional LLM narration sends aggregate interpretation summaries, not uploaded raw rows or customer identifiers. Review your data policy before enabling external processing.")
    st.subheader("Current limitations")
    for x in [
        "This is a local academic prototype with no live Zoho/HubSpot/Salesforce/monday/Pipedrive/ActiveCampaign connection or write actions.",
        "CRM research covers 14 selected capability dimensions (84 platform-capability pairs) rather than every released feature. Sixty-eight rows are intentionally unverified and must not be read as feature absence.",
        "Customer analytics uses rule-based schema suggestions and broad metrics engine, but field definitions, periods, currency and row granularity still require review.",
        "CLV is a historical proxy in this workflow unless a separately validated predictive model is built. An inactivity proxy is not confirmed churn.",
        "Predictive churn-model training, causal price elasticity, formal PDF export, authentication/multi-user storage and production access controls are not implemented in this build.",
    ]: st.markdown(f"- {x}")
