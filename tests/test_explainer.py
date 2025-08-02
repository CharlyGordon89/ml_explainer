import pytest
import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier  # Requires xgboost package
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
    # Add random_state for reproducibility
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(*iris_data)
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


@pytest.mark.parametrize("model_class,expected_output", [
    (RandomForestClassifier, "explanation"),  # Returns Explanation object
    (LogisticRegression, "list"),            # Returns list of arrays
    (MLPClassifier, "list"),                 # Returns list of arrays
    (XGBClassifier, "explanation")           # Returns Explanation object
])


def test_shap_with_multiple_models(iris_data, model_class, expected_output):
    """Verify SHAP works across model architectures"""
    X, y = iris_data
    
    # Special handling for different models
    if model_class == MLPClassifier:
        model = model_class(hidden_layer_sizes=(10,), max_iter=500).fit(X, y)
    elif model_class == LogisticRegression:
        model = model_class(max_iter=1000).fit(X, y)
    else:
        model = model_class().fit(X, y)
    
    explanation = explain_with_shap(model, X)
    
    if expected_output == "explanation":
        assert isinstance(explanation, shap.Explanation)
        # For multi-class, check it has 3 dimensions
        if len(np.unique(y)) > 2:
            assert len(explanation.values.shape) == 3
        else:
            assert len(explanation.values.shape) == 2
    else:
        assert isinstance(explanation, list)
        assert all(isinstance(arr, np.ndarray) for arr in explanation)
        assert len(explanation[0].shape) == 2  # (samples, features)


def test_shap_with_non_sklearn():
    """Test minimal predict() interface with KernelExplainer"""
    class CustomModel:
        def predict(self, X):
            return np.random.rand(len(X))  # Regression output
            
    X = np.random.rand(100, 5)
    model = CustomModel()
    
    # Explicitly use KernelExplainer
    explainer = shap.KernelExplainer(model.predict, X)
    shap_values = explainer.shap_values(X)
    
    assert isinstance(shap_values, np.ndarray)
    assert shap_values.shape == (100, 5)

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


def test_lime_with_regression():
    """Test LIME with regression models"""
    data = load_diabetes()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    model = RandomForestRegressor().fit(X, data.target)
    
    explainer = explain_with_lime(
        model,
        X,
        feature_names=data.feature_names,
        mode="regression"  # Key difference
    )
    
    assert explainer.mode == "regression"

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