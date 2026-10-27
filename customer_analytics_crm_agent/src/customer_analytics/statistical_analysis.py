"""Explicit user-selected statistical tests; no automatic fishing across all columns."""
from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

class StatisticalAnalysisError(ValueError):
    """A test cannot be computed from the selected fields without invalid assumptions."""


def _numeric_pair(df: pd.DataFrame, x: str, y: str) -> pd.DataFrame:
    if x not in df.columns or y not in df.columns:
        raise StatisticalAnalysisError("Choose two columns that exist in the current dataset.")
    if x == y:
        raise StatisticalAnalysisError("Choose two different columns.")
    temp=pd.DataFrame({"x":pd.to_numeric(df[x],errors="coerce"),"y":pd.to_numeric(df[y],errors="coerce")}).replace([np.inf,-np.inf],np.nan).dropna()
    if len(temp)<3:
        raise StatisticalAnalysisError(f"Only {len(temp)} complete numeric pairs are available; at least 3 are required.")
    if temp["x"].nunique()<2 or temp["y"].nunique()<2:
        raise StatisticalAnalysisError("At least one selected numeric variable is constant; the test is undefined.")
    return temp


def run_statistical_test(df: pd.DataFrame, method: str, *, x: str | None = None, y: str | None = None,
                         category_x: str | None = None, category_y: str | None = None) -> dict[str, Any]:
    """Return result objects suitable for report/automatic interpretation."""
    if method == "Pearson correlation":
        if not x or not y: raise StatisticalAnalysisError("Select both numeric columns.")
        temp=_numeric_pair(df,x,y)
        test=stats.pearsonr(temp["x"],temp["y"])
        r=float(test.statistic); p=float(test.pvalue); n=len(temp)
        ci_note=""
        if n>3:
            ci=test.confidence_interval(confidence_level=.95)
            ci_note=f" Fisher-z 95% confidence interval for r: [{ci.low:.3f}, {ci.high:.3f}]."
        direction="positive" if r>0 else "negative"
        strength="very weak" if abs(r)<.2 else "weak" if abs(r)<.4 else "moderate" if abs(r)<.6 else "strong"
        significance="statistically significant at alpha=0.05" if p<.05 else "not statistically significant at alpha=0.05"
        return {"summary":f"Pearson correlation between {x} and {y}: r={r:.3f}, p={p:.4g}, n={n}; {strength} {direction} association; {significance}.{ci_note}",
                "table":pd.DataFrame([{ "Variable X":x,"Variable Y":y,"Pearson r":r,"p-value":p,"Complete pairs":n,"CI 95% lower":float(ci.low) if n>3 else np.nan,"CI 95% upper":float(ci.high) if n>3 else np.nan}]),
                "note":"Correlation measures linear association, not causation. Check scatterplots, influential observations, measurement quality, independence and multiple-testing context before making a decision.",
                "method":"Pearson correlation","sample_size":n,"statistic":r,"p_value":p}
    if method == "Linear regression (one predictor)":
        if not x or not y: raise StatisticalAnalysisError("Select one numeric predictor and one numeric outcome.")
        temp=_numeric_pair(df,x,y)
        if len(temp)<8: raise StatisticalAnalysisError(f"Only {len(temp)} complete pairs are available; this tool requires at least 8 for the single-predictor regression summary.")
        X=sm.add_constant(temp["x"].astype(float),has_constant="add")
        fit=sm.OLS(temp["y"].astype(float),X).fit()
        coef=pd.DataFrame({"Term":["Intercept",x],"Coefficient":[float(fit.params.iloc[0]),float(fit.params.iloc[1])],
                           "Std. error":[float(fit.bse.iloc[0]),float(fit.bse.iloc[1])],
                           "p-value":[float(fit.pvalues.iloc[0]),float(fit.pvalues.iloc[1])],
                           "95% CI lower":[float(fit.conf_int().iloc[0,0]),float(fit.conf_int().iloc[1,0])],
                           "95% CI upper":[float(fit.conf_int().iloc[0,1]),float(fit.conf_int().iloc[1,1])]})
        beta=float(fit.params.iloc[1]); p=float(fit.pvalues.iloc[1]); r2=float(fit.rsquared); n=len(temp)
        sig="statistically distinguishable from zero at alpha=0.05" if p<.05 else "not statistically distinguishable from zero at alpha=0.05"
        return {"summary":f"OLS regression predicting {y} from {x}: slope={beta:.4g}, slope p={p:.4g}, R²={r2:.3f}, n={n}. The slope is {sig}.",
                "table":coef,"note":"Single-predictor OLS is an association model, not proof of cause. Review residuals, nonlinearity, heteroskedasticity, influential observations, confounding, and out-of-sample performance before using it for prediction.",
                "method":"Linear regression (one predictor)","sample_size":n,"coefficient":beta,"p_value":p,"r_squared":r2}
    if method == "Chi-square independence test":
        if not category_x or not category_y: raise StatisticalAnalysisError("Select two categorical columns.")
        if category_x==category_y: raise StatisticalAnalysisError("Choose two different categorical columns.")
        if category_x not in df.columns or category_y not in df.columns: raise StatisticalAnalysisError("Choose categorical columns in the dataset.")
        temp=df[[category_x,category_y]].dropna().astype(str)
        if len(temp)<10: raise StatisticalAnalysisError(f"Only {len(temp)} complete records are available; at least 10 are required for this exploratory test.")
        if temp[category_x].nunique()<2 or temp[category_y].nunique()<2: raise StatisticalAnalysisError("Both categorical variables need at least two observed categories.")
        if temp[category_x].nunique()>20 or temp[category_y].nunique()>20: raise StatisticalAnalysisError("One selected column has more than 20 categories. Select a meaningful lower-cardinality variable; do not test identifiers as categories.")
        table=pd.crosstab(temp[category_x],temp[category_y])
        chi2,p,dof,expected=stats.chi2_contingency(table,correction=False)
        n=int(table.to_numpy().sum()); denom=min(table.shape[0]-1,table.shape[1]-1)
        cramers=float(np.sqrt(chi2/(n*denom))) if denom>0 and n>0 else np.nan
        low_expected=int((expected<5).sum())
        significance="evidence of association at alpha=0.05" if p<.05 else "no statistically significant association at alpha=0.05 detected"
        note=f"{low_expected} of {expected.size} expected cell counts are below 5. Interpret the asymptotic p-value cautiously." if low_expected else "Expected-count screen found no cells below 5; also check independent observations and category definitions."
        return {"summary":f"Chi-square test of independence for {category_x} vs {category_y}: χ²={chi2:.3f}, df={dof}, p={p:.4g}, n={n}; Cramér's V={cramers:.3f}. Results show {significance}.",
                "table":table.reset_index(),"note":note+" Association does not establish causal impact.","method":"Chi-square independence test","sample_size":n,"statistic":float(chi2),"degrees_freedom":int(dof),"p_value":float(p),"cramers_v":cramers,"expected_counts_below_5":low_expected}
    raise StatisticalAnalysisError("Unsupported statistical method. Choose Pearson correlation, linear regression, or chi-square independence test.")
