import pytest
import pandas as pd
import numpy as np
import os
import sys

# Ensure app package is importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app")))

from utils import (
    split_train_test,
    fit_model,
    evaluate,
    predict
)


@pytest.fixture
def sample_feature_matrix():
    """Generates a synthetic feature matrix matching the schema of get_features output."""
    np.random.seed(42)
    n_samples = 50
    return pd.DataFrame({
        "Elo_Home": np.random.uniform(1400, 1600, n_samples),
        "Elo_Away": np.random.uniform(1400, 1600, n_samples),
        "Diff_Elo": np.random.uniform(-100, 100, n_samples),
        "current_form_Home": np.random.randint(0, 10, n_samples),
        "current_form_Away": np.random.randint(0, 10, n_samples),
        "home_team_rest_days": np.random.randint(3, 10, n_samples),
        "away_team_rest_days": np.random.randint(3, 10, n_samples),
        "result_encoded": np.random.choice([0.0, 1.0, 2.0], size=n_samples)
    })


def test_split_train_test(sample_feature_matrix):
    """Test temporal train/test split without shuffling."""
    test_size = 0.2
    X_train, X_test, y_train, y_test = split_train_test(sample_feature_matrix, test_size=test_size)
    
    total = len(sample_feature_matrix)
    expected_test_len = int(total * test_size)
    expected_train_len = total - expected_test_len
    
    assert len(X_train) == expected_train_len
    assert len(X_test) == expected_test_len
    assert len(y_train) == expected_train_len
    assert len(y_test) == expected_test_len
    
    # Ensure shuffle=False (indices are continuous sequential blocks)
    assert list(X_train.index) == list(range(expected_train_len))
    assert list(X_test.index) == list(range(expected_train_len, total))


def test_fit_model_reproducibility(sample_feature_matrix):
    """Test model training and deterministic reproducibility with seed=42."""
    X_train, X_test, y_train, y_test = split_train_test(sample_feature_matrix)
    
    # Train two models with identical parameters
    model1 = fit_model(X_train, y_train, learning_rate=0.05, n_estimators=50, max_depth=3)
    model2 = fit_model(X_train, y_train, learning_rate=0.05, n_estimators=50, max_depth=3)
    
    pred_prob1 = model1.predict_proba(X_test)
    pred_prob2 = model2.predict_proba(X_test)
    
    # Predictions should be identical for fixed seed
    np.testing.assert_array_almost_equal(pred_prob1, pred_prob2)
    np.testing.assert_array_equal(model1.feature_importances_, model2.feature_importances_)


def test_evaluate(sample_feature_matrix):
    """Test model evaluation metric calculation."""
    X_train, X_test, y_train, y_test = split_train_test(sample_feature_matrix)
    model = fit_model(X_train, y_train, n_estimators=20)
    
    results = evaluate(model, X_test, y_test)
    
    assert "log_loss" in results
    assert "accuracy" in results
    assert "y_pred" in results
    assert "y_pred_prob" in results
    
    assert isinstance(results["log_loss"], float)
    assert 0.0 <= results["accuracy"] <= 1.0
    assert len(results["y_pred"]) == len(y_test)


def test_predict(sample_feature_matrix):
    """Test prediction interface and confusion matrix display."""
    X_train, X_test, y_train, y_test = split_train_test(sample_feature_matrix)
    model = fit_model(X_train, y_train, n_estimators=20)
    
    # Test without y_test
    preds = predict(model, X_test)
    assert len(preds) == len(X_test)
    
    # Test with y_test
    preds_cm, cm, disp = predict(model, X_test, y_test)
    assert cm.shape == (3, 3)
    assert disp is not None
