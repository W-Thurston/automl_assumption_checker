# tests/test_linear_model_wrapper.py
import numpy as np
import pandas as pd

from src.models.linear_model_wrapper import LinearModelWrapper


def test_linear_wrapper_fit_and_predict():
    """
    Test that LinearModelWrapper can fit a model and return predictions.
    Ensures the fitted model exposes .predict() and matches expected length.
    """
    X = pd.DataFrame({"x1": np.random.randn(100)})
    y = 3 * X["x1"] + np.random.randn(100)

    model = LinearModelWrapper(X, y).fit()
    preds = model.predict()

    assert len(preds) == len(y)
    assert hasattr(model, "model")


def test_linear_wrapper_residuals_and_fitted():
    """
    Verify that residuals + fitted values approximately equal the true target.
    Confirms internal math and data shape integrity.
    """
    X = pd.DataFrame({"x1": np.random.randn(100)})
    y = 2 * X["x1"] + np.random.randn(100)

    model = LinearModelWrapper(X, y).fit()
    residuals = model.residuals()
    fitted = model.fitted()

    # residuals = y - y_pred
    np.testing.assert_allclose(y.values, residuals + fitted, rtol=1e-4)


def test_linear_wrapper_summary():
    """
    Confirm the summary() method returns expected keys and types.
    """
    X = pd.DataFrame({"x1": np.random.randn(50)})
    y = X["x1"] + np.random.randn(50)

    model = LinearModelWrapper(X, y).fit()
    summary = model.summary()

    assert "model_type" in summary
    assert summary["model_type"].lower() == "linear regression"
    assert 0 <= summary["r_squared"] <= 1
