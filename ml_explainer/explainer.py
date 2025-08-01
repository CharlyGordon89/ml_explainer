# ml_explainer/explainer.py

import shap
import lime.lime_tabular
from sklearn.inspection import permutation_importance
import numpy as np
import pandas as pd


def explain_with_shap(model, X, nsamples=100):
    """Generate SHAP values for a scikit-learn model using a subset of the data."""
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X[:nsamples], check_additivity=False)
    return shap_values


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
