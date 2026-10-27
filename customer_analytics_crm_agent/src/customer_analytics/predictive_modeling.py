"""Optional, evaluation-first churn models for datasets with an explicit binary label.

The function aggregates repeated records to a customer level, rejects customers with
conflicting labels, and evaluates models on a stratified holdout set. It does not infer
churn from inactivity or report a calibrated causal effect.
"""
from __future__ import annotations
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class ChurnModelError(ValueError):
    """Raised when the supplied data cannot support a defensible churn model."""


def _mode_or_missing(values: pd.Series) -> Any:
    vals = values.dropna()
    if vals.empty:
        return np.nan
    modes = vals.mode(dropna=True)
    return modes.iloc[0] if not modes.empty else vals.iloc[0]


def train_churn_models(
    data: pd.DataFrame,
    *,
    customer_id_col: str,
    target_col: str,
    positive_label: str,
    feature_cols: list[str],
    feature_timing_confirmed: bool,
    test_size: float = 0.25,
    random_state: int = 42,
) -> dict[str, Any]:
    """Train/evaluate logistic regression and random forest on a held-out customer sample.

    Selected feature values are aggregated to customer grain (numeric mean; categorical
    mode). This is only suitable where selected features predate the outcome window; the
    caller must explicitly confirm this. Model probabilities are test-set evaluation
    outputs, not claims of calibrated production risk.
    """
    if not feature_timing_confirmed:
        raise ChurnModelError("Confirm that every selected predictor is known before the churn outcome/window.")
    if data.empty or customer_id_col not in data or target_col not in data:
        raise ChurnModelError("A non-empty dataset, customer identifier and explicit churn target are required.")
    features = list(dict.fromkeys(feature_cols))
    if not features:
        raise ChurnModelError("Select at least one predictor column.")
    if customer_id_col in features or target_col in features:
        raise ChurnModelError("Customer ID and target must not be included as predictors.")
    missing = [c for c in features if c not in data.columns]
    if missing:
        raise ChurnModelError("Unknown predictor column(s): " + ", ".join(missing))

    source = data[[customer_id_col, target_col] + features].copy()
    source = source.dropna(subset=[customer_id_col, target_col])
    if source.empty:
        raise ChurnModelError("No records have both a customer ID and target label.")

    # Require exactly two explicit classes; don't silently collapse additional statuses.
    raw_classes = sorted(source[target_col].astype(str).str.strip().unique().tolist())
    positive = str(positive_label).strip()
    if len(raw_classes) != 2:
        raise ChurnModelError(f"A binary target is required, but {len(raw_classes)} label(s) were found: {raw_classes}.")
    if positive not in raw_classes:
        raise ChurnModelError(f"Chosen positive label '{positive}' was not found in the target.")

    # IDs with conflicting churn labels are invalid for customer-level modelling.
    label_counts = source.groupby(customer_id_col, dropna=True)[target_col].nunique(dropna=True)
    conflicting = label_counts[label_counts > 1]
    if not conflicting.empty:
        raise ChurnModelError(f"{len(conflicting)} customer(s) have conflicting churn labels. Resolve the target definition before modelling.")

    # Convert target to binary only after explicit positive-class selection.
    source["__target_binary"] = (source[target_col].astype(str).str.strip() == positive).astype(int)

    # Prevent obvious identifier-like predictors and near-unique text fields from dominating a model.
    n_ids = source[customer_id_col].nunique(dropna=True)
    id_name_hint = __import__("re").compile(r"(^|[_\s-])(id|key|uuid|email|phone|mobile|address|postcode|postal|zip|customer.?name|account.?number)([_\s-]|$)", __import__("re").I)
    likely_ids = []
    for col in features:
        distinct = source[col].nunique(dropna=True)
        is_text = not pd.api.types.is_numeric_dtype(source[col])
        near_unique_text = is_text and n_ids >= 30 and distinct >= max(30, int(0.8 * n_ids))
        if id_name_hint.search(str(col)) or near_unique_text:
            likely_ids.append(col)
    if likely_ids:
        raise ChurnModelError("Predictor(s) appear identifier-like and may cause leakage/overfitting: " + ", ".join(likely_ids) + ". Remove them or justify their use outside this prototype.")

    numeric_features: list[str] = []
    categorical_features: list[str] = []
    for col in features:
        if pd.api.types.is_numeric_dtype(source[col]):
            numeric_features.append(col)
        else:
            categorical_features.append(col)
    # Group to one row per customer. This uses only selected predictors.
    aggregations: dict[str, Any] = {"__target_binary": "first"}
    for col in numeric_features:
        aggregations[col] = "mean"
    for col in categorical_features:
        aggregations[col] = _mode_or_missing
    customer = source.groupby(customer_id_col, dropna=True).agg(aggregations).reset_index()
    customer = customer.dropna(subset=["__target_binary"])

    # Remove all-null and constant predictors, which add no signal and can break imputation.
    numeric_features = [c for c in numeric_features if customer[c].notna().any() and customer[c].nunique(dropna=True) > 1]
    categorical_features = [c for c in categorical_features if customer[c].notna().any() and customer[c].nunique(dropna=True) > 1]
    used_features = numeric_features + categorical_features
    if not used_features:
        raise ChurnModelError("No non-constant predictors with usable values remain after customer-level aggregation.")

    y = customer["__target_binary"].astype(int)
    class_counts = y.value_counts()
    if len(class_counts) != 2 or int(class_counts.min()) < 10 or len(customer) < 40:
        raise ChurnModelError("For this exploratory holdout evaluation, provide at least 40 customers and at least 10 customers in each target class. Use a larger sample for stronger validation.")
    if not 0.15 <= test_size <= 0.4:
        raise ChurnModelError("Test-set proportion must be between 0.15 and 0.40.")

    X = customer[used_features].copy()
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, customer.index, test_size=test_size, random_state=random_state, stratify=y
    )
    transformers = []
    if numeric_features:
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ])
        transformers.append(("numeric", numeric_pipeline, numeric_features))
    if categorical_features:
        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ])
        transformers.append(("categorical", categorical_pipeline, categorical_features))
    preprocessor = ColumnTransformer(transformers, remainder="drop")

    model_specs = {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_state),
        "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced_subsample", random_state=random_state, n_jobs=1),
    }
    rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    prediction = pd.DataFrame({
        "Customer ID": customer.loc[idx_test, customer_id_col].astype(str).to_numpy(),
        "Actual target label": np.where(y_test.to_numpy() == 1, positive, next(c for c in raw_classes if c != positive)),
    })
    class_names = {1: positive, 0: next(c for c in raw_classes if c != positive)}
    confusion_tables: dict[str, pd.DataFrame] = {}
    for model_name, classifier in model_specs.items():
        pipe = Pipeline([("preprocess", preprocessor), ("model", classifier)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        prob = pipe.predict_proba(X_test)[:, list(pipe.named_steps["model"].classes_).index(1)]
        auc = float(roc_auc_score(y_test, prob)) if y_test.nunique() == 2 else np.nan
        rows.append({
            "Model": model_name,
            "Customers (total)": int(len(customer)),
            "Train customers": int(len(X_train)),
            "Test customers": int(len(X_test)),
            "Accuracy": float(accuracy_score(y_test, pred)),
            "Balanced accuracy": float(balanced_accuracy_score(y_test, pred)),
            "Precision (positive class)": float(precision_score(y_test, pred, zero_division=0)),
            "Recall (positive class)": float(recall_score(y_test, pred, zero_division=0)),
            "F1 (positive class)": float(f1_score(y_test, pred, zero_division=0)),
            "ROC AUC": auc,
        })
        prediction[f"{model_name} predicted label"] = np.where(pred == 1, positive, class_names[0])
        prediction[f"{model_name} positive-class score"] = prob
        cm = confusion_matrix(y_test, pred, labels=[0, 1])
        confusion_tables[model_name] = pd.DataFrame(cm, index=[f"Actual {class_names[0]}", f"Actual {class_names[1]}"], columns=[f"Predicted {class_names[0]}", f"Predicted {class_names[1]}"]).reset_index(names="Actual / predicted")
        fitted_transform = pipe.named_steps["preprocess"]
        try:
            transformed_names = fitted_transform.get_feature_names_out()
            fitted_model = pipe.named_steps["model"]
            if model_name == "Logistic Regression":
                values = fitted_model.coef_[0]
                for name, value in zip(transformed_names, values):
                    feature_rows.append({"Model": model_name, "Feature": str(name), "Importance / coefficient": float(value), "Absolute magnitude": abs(float(value)), "Interpretation": "Positive/negative association with the model's positive class; not causal."})
            else:
                values = fitted_model.feature_importances_
                for name, value in zip(transformed_names, values):
                    feature_rows.append({"Model": model_name, "Feature": str(name), "Importance / coefficient": float(value), "Absolute magnitude": abs(float(value)), "Interpretation": "Split-based importance; not causal."})
        except (AttributeError, ValueError):
            pass
    metrics = pd.DataFrame(rows).sort_values("ROC AUC", ascending=False, na_position="last").reset_index(drop=True)
    feature_table = pd.DataFrame(feature_rows)
    if not feature_table.empty:
        feature_table = feature_table.sort_values(["Model", "Absolute magnitude"], ascending=[True, False]).groupby("Model", sort=False).head(15).reset_index(drop=True)
    summary = (
        f"Evaluated logistic regression and random forest on {len(customer)} unique customers with an explicit churn target; "
        f"stratified holdout test size was {len(X_test)} ({len(X_test)/len(customer):.1%}). Positive class: '{positive}'. "
        f"Best holdout ROC AUC: {metrics.iloc[0]['ROC AUC']:.3f} ({metrics.iloc[0]['Model']})."
    )
    note = ("Exploratory holdout estimates, not a validated production churn model or calibrated probability of churn. Selected predictors are aggregated to customer level (numeric mean; categorical mode); this is defensible only if all predictors precede the churn window. "
            "Use time-based validation for deployment, check leakage, drift, calibration, subgroup performance and intervention capacity. Precision/recall depend on the selected positive class and threshold; a high score does not prove a customer will churn.")
    return {
        "summary": summary, "note": note, "method": "Exploratory churn model comparison", "sample_size": int(len(customer)),
        "metrics": metrics, "feature_importance": feature_table, "predictions": prediction,
        "confusion_matrices": confusion_tables, "positive_label": positive,
        "negative_label": next(c for c in raw_classes if c != positive),
        "features": used_features, "customer_level_dataset": customer,
        "model_selection_warning": "Choose the final model using business-relevant metrics, not accuracy alone; validate on a later period before production use.",
    }
