import shap
import lime.lime_tabular
from sklearn.inspection import permutation_importance
import numpy as np
import pandas as pd
from typing import Union, List
from sklearn.base import BaseEstimator
from typing import Any
import warnings

def explain_with_shap(
    model: Union[BaseEstimator, Any],
    X: Union[pd.DataFrame, np.ndarray],
    nsamples: int = 100,
    check_additivity: bool = False,
    approximate: bool = True,
    **shap_kwargs
) -> Union[shap.Explanation, list]:
    """
    Generate SHAP values with robust model compatibility handling.
    Returns either:
    - shap.Explanation object (for most models)
    - list of arrays (for multi-class classifiers when using KernelExplainer)
    """
    # Input validation
    if not hasattr(model, 'predict'):
        raise ValueError("Model must implement predict() method")
    
    if not isinstance(X, (pd.DataFrame, np.ndarray)):
        raise TypeError(f"X must be DataFrame or ndarray, got {type(X)}")

    try:
        # Handle tree-based models
        if hasattr(model, 'tree_') or 'XGB' in str(type(model)) or 'GradientBoosting' in str(type(model)):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X[:nsamples], 
                                              check_additivity=check_additivity,
                                              approximate=approximate)
            
            # Handle multi-class output
            if isinstance(shap_values, list):
                return shap.Explanation(
                    values=np.stack(shap_values, axis=-1),
                    base_values=np.array(explainer.expected_value),
                    data=X[:nsamples],
                    feature_names=X.columns if hasattr(X, 'columns') else None
                )
            return shap_values
            
        # Handle linear models
        elif 'linear_model' in str(type(model)):
            explainer = shap.LinearExplainer(model, X)
            shap_values = explainer.shap_values(X[:nsamples])
            # LinearExplainer returns list for multi-class
            if isinstance(shap_values, list):
                return shap_values
            return shap.Explanation(
                values=shap_values,
                base_values=explainer.expected_value,
                data=X[:nsamples],
                feature_names=X.columns if hasattr(X, 'columns') else None
            )
            
        # Handle neural networks
        elif 'MLP' in str(type(model)):
            background = shap.sample(X, 50)
            explainer = shap.KernelExplainer(model.predict_proba, background)
            return explainer.shap_values(X[:nsamples])
            
        # Default explainer for other cases
        else:
            explainer = shap.Explainer(model, X)
            result = explainer(X[:nsamples], 
                             check_additivity=check_additivity,
                             **shap_kwargs)
            return result if isinstance(result, list) else result
            
    except Exception as e:
        warnings.warn(f"SHAP explanation failed: {str(e)}")
        raise


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
    """Parallel computation of SHAP values. 
       Note: May not work with all explainer types.
       Returns:
            shap.Explanation object   
    """
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