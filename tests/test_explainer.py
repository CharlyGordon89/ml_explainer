import pytest
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.datasets import load_iris, load_diabetes

from ml_explainer.explainer import (
    explain_with_shap,
    explain_with_lime,
    explain_with_permutation
)

# --------------------------
# Fixtures
# --------------------------
@pytest.fixture
def iris_data():
    data = load_iris()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target
    return X, y

@pytest.fixture
def iris_model(iris_data):
    X, y = iris_data
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    return model

@pytest.fixture
def shap_values_fixture():
    X = np.random.rand(100, 5)
    y = np.random.randint(0, 2, 100)
    model = RandomForestClassifier().fit(X, y)
    return explain_with_shap(model, X)

# --------------------------
# SHAP Tests
# --------------------------
def test_shap_explainer_with_array(shap_values_fixture):
    """Test with numpy array input"""
    assert len(shap_values_fixture) == min(100, 100)  # 100 is hardcoded in fixture
    assert hasattr(shap_values_fixture, 'values')

def test_shap_explainer_with_dataframe(iris_data, iris_model):
    """Test with DataFrame input"""
    X_df, _ = iris_data
    shap_values_df = explain_with_shap(iris_model, X_df, approximate=True)
    assert isinstance(shap_values_df, shap.Explanation)
    assert shap_values_df.shape[0] <= 100

def test_shap_kwargs_forwarding():
    """Test SHAP parameter passthrough"""
    X = pd.DataFrame(np.random.rand(50, 3), columns=["f1", "f2", "f3"])
    model = RandomForestClassifier().fit(X, np.random.randint(0, 2, 50))
    with pytest.raises(TypeError):
        explain_with_shap(model, X, invalid_param=True)

# --------------------------
# LIME Tests
# --------------------------
def test_explain_with_lime(iris_data, iris_model):
    """Test LIME explainer creation"""
    X, y = iris_data
    data = load_iris()
    
    lime_explainer = explain_with_lime(
        iris_model,
        X,
        feature_names=data.feature_names,
        class_names=data.target_names.tolist(),
        mode="classification"
    )
    assert lime_explainer is not None
    assert lime_explainer.feature_names == list(data.feature_names)


# --------------------------
# Permutation Importance Tests
# --------------------------
def test_explain_with_permutation(iris_data):
    """Basic permutation importance test with all features"""
    X, y = iris_data
    model = RandomForestClassifier().fit(X, y)  # Train on full feature set
    importance_df = explain_with_permutation(model, X, y)
    
    assert not importance_df.empty
    assert "feature" in importance_df.columns
    assert "importance_mean" in importance_df.columns
    assert len(importance_df) == 4  # All iris features

def test_permutation_with_custom_scoring():
    """Test alternate scoring metrics with proper feature handling"""
    data = load_diabetes()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target
    model = RandomForestRegressor().fit(X, y)
    
    importance_df = explain_with_permutation(
        model, X, y, 
        scoring="neg_mean_squared_error"
    )
    assert "importance_mean" in importance_df.columns
    assert len(importance_df) == X.shape[1]  # All diabetes features

@pytest.mark.parametrize("n_features", [1, 2, 3])  # Tests multiple subset sizes
def test_permutation_with_feature_subsets(iris_data, n_features):
    """Test partial feature sets with properly trained models"""
    X, y = iris_data
    X_subset = X.iloc[:, :n_features]
    model = RandomForestClassifier().fit(X_subset, y)  # Key: Train on subset
    
    importance_df = explain_with_permutation(model, X_subset, y)
    assert len(importance_df) == n_features
    assert all(f in X_subset.columns for f in importance_df["feature"])

def test_permutation_with_mismatched_features(iris_data):
    """Verify proper error when features don't match training"""
    X, y = iris_data
    model = RandomForestClassifier().fit(X, y)  # Trained on full features
    
    with pytest.raises(ValueError, match="feature names should match"):
        # Attempt explanation with subset of features
        explain_with_permutation(model, X.iloc[:, :2], y)