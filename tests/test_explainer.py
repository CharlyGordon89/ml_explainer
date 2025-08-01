# tests/test_explainer.py

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from ml_explainer.explainer import (
    explain_with_shap,
    explain_with_lime,
    explain_with_permutation
)


def test_explain_with_shap():
    data = load_iris()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target

    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)

    shap_values = explain_with_shap(model, X)
    assert hasattr(shap_values, "values")
    assert shap_values.shape[0] <= 100  # nsamples default


def test_explain_with_lime():
    data = load_iris()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target

    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)

    lime_explainer = explain_with_lime(
        model,
        X,
        feature_names=data.feature_names,
        class_names=data.target_names.tolist(),
        mode="classification"
    )

    assert lime_explainer is not None
    assert lime_explainer.feature_names == list(data.feature_names)


def test_explain_with_permutation():
    data = load_iris()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target

    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)

    importance_df = explain_with_permutation(model, X, y)
    assert not importance_df.empty
    assert "feature" in importance_df.columns
    assert "importance_mean" in importance_df.columns
