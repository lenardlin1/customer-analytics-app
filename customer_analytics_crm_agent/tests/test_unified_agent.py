from __future__ import annotations
import io, json, sys
from pathlib import Path
import pandas as pd
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from customer_analytics.crm_intelligence import load_knowledge_base, validate_knowledge_base, compare_capabilities, assess_gap, search_capabilities, FEATURES
from customer_analytics.full_analytics import FIELD_DEFS, NOT_MAPPED, guess_column, compute_metrics, quality_report, feasibility, apply_cleaning
from customer_analytics.automatic_interpretation import interpret_all


def test_kb_has_84_pairs_16_verified_and_valid_schema():
    kb=load_knowledge_base(ROOT/'data'/'crm_capabilities.json')
    assert len(kb['capabilities'])==84
    assert sum(x['verification_status']=='verified' for x in kb['capabilities'])==16
    assert len(validate_knowledge_base(kb))==0
    assert len(kb['platforms'])==6


def test_unverified_never_turns_into_confirmed_functional_gap():
    res=assess_gap({'evidence_verified':False,'availability_status':'not_verified','configuration_status':'unavailable_confirmed',
      'data_available':True,'integration_needed':False,'process_issue':False,'impact':5,'frequency':5,'manual_effort':5,
      'customer_financial_consequence':5,'differentiation':5,'feasibility':5,'customer_analytics_relevance':5,'evidence_confidence':1})
    assert res['gap_type']=='Evidence/research gap'
    assert res['confirmed_finding'] is False


def test_documented_configuration_gap_is_classified_conservatively():
    res=assess_gap({'evidence_verified':True,'availability_status':'verified_available','configuration_status':'not_configured',
      'data_available':True,'integration_needed':False,'process_issue':False,'impact':4,'frequency':4,'manual_effort':3,
      'customer_financial_consequence':4,'differentiation':3,'feasibility':4,'customer_analytics_relevance':5,'evidence_confidence':4})
    assert res['gap_type']=='Configuration/adoption gap'
    assert res['confirmed_finding'] is True


def test_comparison_returns_not_verified_for_unresearched_pair():
    kb=load_knowledge_base(ROOT/'data'/'crm_capabilities.json')
    # monday CRM custom data model pair remains explicitly unverified
    row=compare_capabilities(['monday CRM'],['custom_data_model'],data=kb)[0]
    assert row['status']=='not_verified'
    assert 'do not infer unavailability' in row['limitations']


def test_search_churn_surfaces_verified_churn_evidence():
    kb=load_knowledge_base(ROOT/'data'/'crm_capabilities.json')
    results=search_capabilities('predict customer churn', ['Zoho CRM'], data=kb)
    assert results
    assert any(r['verification_status']=='verified' and r['feature_id']=='health_churn' for r in results)


def test_nps_calculation_known_value():
    df=pd.DataFrame({'NPS_Score':[10,9,8,7,6,0]})
    mapping={k:NOT_MAPPED for k in FIELD_DEFS}; mapping['nps_score']='NPS_Score'
    results,_=compute_metrics(df,mapping,{})
    result=results['Net Promoter Score (NPS)']
    assert 'NPS: 0.0' in result['summary']
    assert 'Promoters 33.3%' in result['summary']
    assert 'detractors 33.3%' in result['summary']


def test_rfm_and_historical_value_with_known_data():
    df=pd.DataFrame({'Customer_ID':['A','A','B'],'Order_ID':['o1','o2','o3'],
      'Transaction_Date':['2026-01-01','2026-01-10','2026-01-05'],'Revenue':[100,150,50]})
    mapping={k:NOT_MAPPED for k in FIELD_DEFS}
    mapping.update({'customer_id':'Customer_ID','order_id':'Order_ID','txn_date':'Transaction_Date','amount':'Revenue'})
    results,_=compute_metrics(df,mapping,{})
    assert 'RFM Analysis' in results
    assert 'Customer Lifetime Value (historical proxy)' in results
    t=results['Customer Lifetime Value (historical proxy)']['table']
    a=t[t['Customer ID']=='A'].iloc[0]
    assert a['Observed revenue']==250


def test_csat_uses_configured_threshold_and_denominator():
    df=pd.DataFrame({'CSAT_Score':[5,4,3,1,None]})
    mapping={k:NOT_MAPPED for k in FIELD_DEFS}; mapping['csat_score']='CSAT_Score'
    results,_=compute_metrics(df,mapping,{'csat_threshold':4})
    assert 'Customer Satisfaction (CSAT)' in results
    assert 'Share at or above the configured threshold (4): 50.0%' in results['Customer Satisfaction (CSAT)']['summary']
    assert 'n=4' in results['Customer Satisfaction (CSAT)']['summary']


def test_cac_not_feasible_when_required_spend_or_acquisition_count_missing():
    df=pd.DataFrame({'Customer_ID':['A','B'],'Revenue':[100,150]})
    mapping={k:NOT_MAPPED for k in FIELD_DEFS}; mapping['customer_id']='Customer_ID'; mapping['amount']='Revenue'
    feas=feasibility(mapping,df,{"manual_spend":None,"manual_new_customers":None})
    row=feas[feas['Analysis']=='Customer Acquisition Cost (CAC)'].iloc[0]
    assert row['Status']=='Not calculable from current mapping'
    assert 'newly acquired customers' in row['Reason / requirement']


def test_exact_duplicates_reported_and_original_not_mutated():
    df=pd.DataFrame({'Customer_ID':['A','A','A'],'Order_ID':['1','1','2'],'Revenue':[50,50,100]})
    before=df.copy(deep=True)
    missing,outliers=quality_report(df)
    cleaned,audit=apply_cleaning(df,False,'Flag only (recommended)')
    assert int(df.duplicated().sum())==1
    pd.testing.assert_frame_equal(df,before)
    assert len(cleaned)==3
    assert any('retained' in x for x in audit)


def test_interpretation_generated_from_computed_output():
    df=pd.DataFrame({'NPS_Score':[10,10,9,7,6,0]})
    mapping={k:NOT_MAPPED for k in FIELD_DEFS}; mapping['nps_score']='NPS_Score'
    results,_=compute_metrics(df,mapping,{})
    interpretations=interpret_all(results,dataset_rows=len(df))
    item=interpretations['Net Promoter Score (NPS)']
    assert item['key_findings']
    assert item['recommendations']
    assert item['limitations']
    assert any('NPS: 16.7' in text for text in item['key_findings'])


def test_demo_data_is_synthetic_and_repeat_transactions_are_preserved():
    df=pd.read_csv(ROOT/'data'/'synthetic_customer_transactions.csv')
    assert len(df)>500
    assert df['Customer_ID'].nunique()>=100
    assert int(df.duplicated().sum())==1
    assert df['Order_ID'].nunique()<len(df) # duplicate row is an exact repeated order row


def test_pearson_correlation_matches_perfect_linear_data():
    from customer_analytics.statistical_analysis import run_statistical_test
    df=pd.DataFrame({'x':[1,2,3,4,5],'y':[2,4,6,8,10]})
    result=run_statistical_test(df,'Pearson correlation',x='x',y='y')
    assert result['statistic']==pytest.approx(1.0)
    assert result['p_value'] < 1e-10
    assert result['sample_size']==5


def test_linear_regression_matches_known_slope():
    from customer_analytics.statistical_analysis import run_statistical_test
    df=pd.DataFrame({'spend':[1,2,3,4,5,6,7,8],'sales':[3,5,7,9,11,13,15,17]})
    result=run_statistical_test(df,'Linear regression (one predictor)',x='spend',y='sales')
    assert result['coefficient']==pytest.approx(2.0)
    assert result['r_squared']==pytest.approx(1.0)


def test_chi_square_reports_counts_effect_size_and_small_expected_cell_caution():
    from customer_analytics.statistical_analysis import run_statistical_test
    df=pd.DataFrame({'channel':['A']*10+['B']*10,'converted':['Y']*8+['N']*2+['Y']*2+['N']*8})
    result=run_statistical_test(df,'Chi-square independence test',category_x='channel',category_y='converted')
    assert result['sample_size']==20
    assert result['p_value'] < .05
    assert result['cramers_v'] > .5
    assert isinstance(result['table'],pd.DataFrame)


def test_stat_test_refuses_constant_numeric_field():
    from customer_analytics.statistical_analysis import run_statistical_test, StatisticalAnalysisError
    with pytest.raises(StatisticalAnalysisError):
        run_statistical_test(pd.DataFrame({'x':[1,1,1,1],'y':[1,2,3,4]}),'Pearson correlation',x='x',y='y')


def test_multisheet_excel_ingestion():
    from customer_analytics.full_analytics import read_uploaded_file
    class Adapter:
        name='test_workbook.xlsx'
        def __init__(self,data): self.data=data
        def getvalue(self): return self.data
    buff=io.BytesIO()
    with pd.ExcelWriter(buff,engine='openpyxl') as writer:
        pd.DataFrame({'Customer_ID':['A'],'Revenue':[100]}).to_excel(writer,index=False,sheet_name='Customers')
        pd.DataFrame({'Order_ID':['O1'],'Revenue':[100]}).to_excel(writer,index=False,sheet_name='Orders')
    adapter=Adapter(buff.getvalue())
    orders,sheets=read_uploaded_file(adapter,'Orders')
    assert sheets==['Customers','Orders']
    assert orders.iloc[0]['Order_ID']=='O1'


def test_synthetic_dataset_runs_multiple_metrics_and_interpretations():
    df=pd.read_csv(ROOT/'data'/'synthetic_customer_transactions.csv')
    mapping={k:guess_column(df,aliases) for k,(label,aliases) in FIELD_DEFS.items()}
    results,meta=compute_metrics(df,mapping,{'inactivity_days':90,'csat_threshold':4,'n_clusters':3})
    interpretations=interpret_all(results,dataset_rows=len(df))
    assert len(results)>=10
    assert set(results)==set(interpretations)
    assert all(v.get('recommendations') and v.get('limitations') for v in interpretations.values())
    assert 'Customer Acquisition Cost (CAC)' not in results # no aligned spend/acquisition count was supplied
    assert 'RFM Analysis' in results
    assert 'Customer Churn (inactivity proxy)' in results


def test_aov_sums_order_lines_before_averaging_orders():
    df=pd.DataFrame({'Order_ID':['O1','O1','O2'],'Revenue':[100,50,200]})
    mapping={k:NOT_MAPPED for k in FIELD_DEFS}; mapping.update({'order_id':'Order_ID','amount':'Revenue'})
    results,_=compute_metrics(df,mapping,{'row_granularity':'One order line / item per row'})
    assert 'Average Order Value (AOV)' in results
    assert '175.00' in results['Average Order Value (AOV)']['summary']
    assert '2 distinct orders' in results['Average Order Value (AOV)']['summary']


def test_aov_refuses_duplicate_order_rows_if_declared_order_level():
    df=pd.DataFrame({'Order_ID':['O1','O1','O2'],'Revenue':[100,100,200]})
    mapping={k:NOT_MAPPED for k in FIELD_DEFS}; mapping.update({'order_id':'Order_ID','amount':'Revenue'})
    results,_=compute_metrics(df,mapping,{'row_granularity':'One transaction / order per row'})
    assert 'Average Order Value (AOV)' in results
    assert 'AOV was not calculated' in results['Average Order Value (AOV)']['summary']


def test_cac_source_columns_require_explicit_grain_and_period_confirmation():
    df=pd.DataFrame({'Spend':[100,100,200],'New_Customers':[1,0,1]})
    mapping={k:NOT_MAPPED for k in FIELD_DEFS}; mapping.update({'acquisition_spend':'Spend','new_customers_count':'New_Customers'})
    not_confirmed,_=compute_metrics(df,mapping,{'cac_inputs_confirmed':False})
    assert 'Customer Acquisition Cost (CAC)' in not_confirmed
    assert 'not calculated' in not_confirmed['Customer Acquisition Cost (CAC)']['summary'].lower()
    confirmed,_=compute_metrics(df,mapping,{'cac_inputs_confirmed':True})
    assert 'Customer Acquisition Cost (CAC)' in confirmed
    assert '200.00' in confirmed['Customer Acquisition Cost (CAC)']['summary']


def test_gap_priority_weights_are_user_configurable_and_reject_all_zero():
    payload={'evidence_verified':False,'availability_status':'not_verified','configuration_status':'unknown',
      'impact':1,'frequency':5,'manual_effort':5,'customer_financial_consequence':5,'differentiation':5,
      'feasibility':5,'customer_analytics_relevance':5,'evidence_confidence':1,
      'weights':{'impact':1,'frequency':0,'manual_effort':0,'customer_financial_consequence':0,'differentiation':0,'feasibility':0,'customer_analytics_relevance':0}}
    result=assess_gap(payload)
    assert result['priority_score']==20
    with pytest.raises(ValueError,match='At least one'):
        assess_gap({**payload,'weights':{k:0 for k in payload['weights']}})


def test_gap_recommendation_contains_owner_actions_kpis_and_measurement_plan():
    result=assess_gap({'evidence_verified':False,'availability_status':'not_verified','configuration_status':'unknown',
      'impact':4,'frequency':3,'manual_effort':3,'customer_financial_consequence':4,'differentiation':3,
      'feasibility':3,'customer_analytics_relevance':5})
    assert result['affected_stakeholder']
    assert len(result['proposed_improvement'])>=1
    assert len(result['kpis_to_monitor'])>=1
    assert result['measurement_plan']
    assert result['recommended_next_step']


def test_predictive_churn_model_compares_two_models_on_holdout_and_returns_interpretation_inputs():
    from customer_analytics.predictive_modeling import train_churn_models
    from customer_analytics.automatic_interpretation import interpret_all
    rng=__import__('numpy').random.RandomState(7)
    n=160
    risk=rng.normal(size=n)
    tenure=rng.randint(1,60,size=n)
    churn=__import__('numpy').where(risk + rng.normal(scale=0.7,size=n)>0.55,"Churn","Retained")
    df=pd.DataFrame({'Customer_ID':[f'C{i:03d}' for i in range(n)],'Churn_Status':churn,'Risk_Feature':risk,'Tenure_Months':tenure,
                     'Preferred_Channel':['Email' if i%3==0 else 'Store' if i%3==1 else 'Web' for i in range(n)]})
    result=train_churn_models(df,customer_id_col='Customer_ID',target_col='Churn_Status',positive_label='Churn',
                              feature_cols=['Risk_Feature','Tenure_Months','Preferred_Channel'],feature_timing_confirmed=True)
    assert set(result['metrics']['Model'])=={'Logistic Regression','Random Forest'}
    assert result['metrics']['ROC AUC'].between(0,1).all()
    assert len(result['predictions'])==int(result['metrics']['Test customers'].iloc[0]) and len(result['predictions'])>0
    interpreted=interpret_all({'Predictive Churn Model Evaluation':{'summary':result['summary'],'table':result['metrics'],'note':result['note']}},dataset_rows=result['sample_size'])
    assert 'held-out ROC AUC' in ' '.join(interpreted['Predictive Churn Model Evaluation']['key_findings'])
    assert 'calibrated production probability' in ' '.join(interpreted['Predictive Churn Model Evaluation']['limitations'])


def test_predictive_churn_requires_explicit_feature_timing_confirmation():
    from customer_analytics.predictive_modeling import train_churn_models, ChurnModelError
    df=pd.DataFrame({'Customer_ID':[f'C{i}' for i in range(50)],'Churn':['Y']*20+['N']*30,'Engagement':list(range(50))})
    with pytest.raises(ChurnModelError,match='before the churn'):
        train_churn_models(df,customer_id_col='Customer_ID',target_col='Churn',positive_label='Y',feature_cols=['Engagement'],feature_timing_confirmed=False)


def test_predictive_churn_rejects_conflicting_customer_outcomes():
    from customer_analytics.predictive_modeling import train_churn_models, ChurnModelError
    df=pd.DataFrame({'Customer_ID':['A','A']+[f'C{i}' for i in range(38)],'Churn':['Y','N']+['Y']*18+['N']*20,'Engagement':list(range(40))})
    with pytest.raises(ChurnModelError,match='conflicting churn labels'):
        train_churn_models(df,customer_id_col='Customer_ID',target_col='Churn',positive_label='Y',feature_cols=['Engagement'],feature_timing_confirmed=True)


def test_excel_export_contains_data_dictionary_audit_interpretations_and_model_auxiliary_tables():
    from customer_analytics.exporting import excel_bundle
    original=pd.DataFrame({'Customer_ID':['A','A','B'],'Revenue':[10,10,999]})
    missing,outliers=quality_report(original)
    cleaned=original.drop_duplicates().copy()
    feas=pd.DataFrame([{'Analysis':'CAC','Status':'Not feasible'}])
    results={'Example':{'summary':'Computed value 12','note':'Example note','table':pd.DataFrame({'x':[1,2]}),
                        'auxiliary_tables':{'Confusion Matrix':pd.DataFrame({'Observed':['A'],'Predicted':['A']})}}}
    interpretations={'Example':{'overview':'Observed','key_findings':['value 12'],'business_implications':'Review it',
        'root_cause_assessment':'Hypothesis','recommendations':['Check source'],'kpis':['KPI'],'limitations':['Demo only']}}
    duplicate_rows=original.loc[original.duplicated(keep=False)].copy()
    blob=excel_bundle(original,cleaned,missing,outliers,feas,results,{'example':'yes'},duplicate_rows=duplicate_rows,
                      outlier_records=pd.DataFrame({'Source_Row_Number':[4],'Column':['Revenue']}),audit=['Duplicate retained'],interpretations=interpretations)
    assert blob[:2]==b'PK'
    with pd.ExcelFile(io.BytesIO(blob)) as book:
        required={'Data Dictionary','Outlier Row Audit','Exact Duplicate Audit','Cleaning Audit','Interpretations','Results Summary','A01_01_Confusion Matrix'}
        assert required.issubset(set(book.sheet_names))
        interp=pd.read_excel(book,sheet_name='Interpretations')
        assert 'Check source' in interp.loc[0,'Recommendations']

def test_outlier_row_report_is_traceable_and_does_not_mutate_input():
    from customer_analytics.full_analytics import outlier_record_report
    df=pd.DataFrame({'Customer_ID':['A','B','C','D','E','F'],'Amount':[10,11,10,12,9,10000]})
    before=df.copy(deep=True)
    report=outlier_record_report(df,'Flag only (recommended)')
    assert len(report)>=1
    assert (report['Column']=='Amount').any()
    assert (report['Source_Row_Number']==7).any()
    assert not report['Change_Applied'].any()
    pd.testing.assert_frame_equal(df,before)


def test_salesforce_prediction_builder_retirement_caveat_overrides_old_feature_docs():
    kb=load_knowledge_base(ROOT/'data'/'crm_capabilities.json')
    row=next(r for r in kb['capabilities'] if r['platform']=='Salesforce' and r['feature_id']=='retention_actions')
    assert row['verification_status']=='verified'
    assert row['availability_status']=='available_plan_restrictions'
    assert 'retired' in row['plan_scope'].lower() or 'retiring' in row['plan_scope'].lower()
    assert '2026-10-03' in row['evidence_note']
    assert row['additional_evidence_urls']
    assert len(validate_knowledge_base(kb))==0


def test_bundled_synthetic_churn_dataset_supports_model_evaluation():
    from customer_analytics.predictive_modeling import train_churn_models
    df=pd.read_csv(ROOT/'data'/'synthetic_churn_modeling.csv')
    assert len(df)==320 and df['Customer_ID'].nunique()==320
    assert set(df['Churn_Within_Next_90_Days'])=={'Churn','Retained'}
    result=train_churn_models(df,customer_id_col='Customer_ID',target_col='Churn_Within_Next_90_Days',positive_label='Churn',
        feature_cols=['Tenure_Months','Monthly_Spend','Engagement_Score','Support_Tickets_90d','Contract_Type','Acquisition_Channel'],feature_timing_confirmed=True)
    assert len(result['metrics'])==2
    assert result['metrics']['ROC AUC'].between(0,1).all()
    assert len(result['predictions'])==80


def test_prioritization_rejects_negative_weights():
    payload={'evidence_verified':False,'availability_status':'not_verified','configuration_status':'unknown',
      'impact':4,'frequency':4,'manual_effort':3,'customer_financial_consequence':4,'differentiation':3,
      'feasibility':3,'customer_analytics_relevance':5,'weights':{'impact':-1,'frequency':1}}
    with pytest.raises(ValueError,match='non-negative'):
        assess_gap(payload)


def test_visualization_builder_uses_actual_retention_nps_and_model_tables():
    from customer_analytics.visualizations import build_analysis_chart
    retention=pd.DataFrame({'Cohort month':['2026-01','2026-02'],'Month 0':[100,100],'Month 1':[55,42]})
    assert build_analysis_chart('Customer Retention',{'table':retention}) is not None
    nps=pd.DataFrame({'Group':['Promoters (9–10)','Passives (7–8)','Detractors (0–6)'],'Count':[20,30,50],'Percent':[20.,30.,50.]})
    assert build_analysis_chart('Net Promoter Score (NPS)',{'table':nps}) is not None
    metrics=pd.DataFrame({'Model':['Logistic Regression','Random Forest'],'ROC AUC':[.72,.68],
                          'Recall (positive class)':[.7,.5],'Precision (positive class)':[.4,.6]})
    assert build_analysis_chart('Predictive Churn Model Evaluation',{'table':metrics}) is not None
