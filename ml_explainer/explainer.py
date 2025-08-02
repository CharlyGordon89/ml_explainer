import shap
import lime.lime_tabular
from sklearn.inspection import permutation_importance
import numpy as np
import pandas as pd
from typing import Union, List
from sklearn.base import BaseEstimator
from typing import Any


def explain_with_shap(
    model: Union[BaseEstimator, Any],
    X: Union[pd.DataFrame, np.ndarray],
    nsamples: int = 100,
    check_additivity=False,
    approximate=True,
    **shap_kwargs) -> shap.Explanation:
    """
    Generate SHAP values with model compatibility checks.
    """
    if not (hasattr(model, 'predict') or hasattr(model, 'predict_proba')):
        warnings.warn(
            "Model may not be SHAP-compatible - missing predict() or predict_proba()",
            UserWarning
        )
    
    if not isinstance(X, (pd.DataFrame, np.ndarray)):
        raise TypeError(f"X must be DataFrame or ndarray, got {type(X)}")
    
    explainer = shap.Explainer(model, X)
    return explainer(
        X[:nsamples],
        check_additivity=check_additivity,
        **shap_kwargs
    )


def explain_with_lime(model, X, feature_names, class_names=None, mode="classification"):
    """Generate LIME explainer object for tabular data."""
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=np.array(X),
        feature_names=feature_names,
        class_names=class_names,
        mode=mode
    )
    return explainer


def explain_with_permutation(model, X, y, scoring="accuracy", n_repeats=10, random_state=42):
    """Compute permutation feature importance for a model."""
    result = permutation_importance(model, X, y, scoring=scoring, n_repeats=n_repeats, random_state=random_state)
    importance_df = pd.DataFrame({
        "feature": X.columns,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std
    }).sort_values(by="importance_mean", ascending=False)
    return importance_df




def explain_with_shap_parallel(model, X, nsamples=100, n_jobs=-1):
    """Parallel computation of SHAP values with proper joblib implementation"""
    explainer = shap.Explainer(model, X)
    
    # Process samples in parallel batches
    results = Parallel(n_jobs=n_jobs)(
        delayed(explainer)(X[i:i+1], check_additivity=False)
        for i in range(min(nsamples, len(X)))
    )
    
    # Combine results into single Explanation object
    return shap.Explanation(
        values=np.vstack([r.values for r in results]),
        base_values=np.vstack([r.base_values for r in results]),
        data=X[:nsamples],
        feature_names=X.columns if hasattr(X, 'columns') else None
    )