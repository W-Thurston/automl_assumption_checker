# tests/test_multicollinearity.py
import numpy as np
import pandas as pd

from src.config import VIF_THRESHOLD
from src.core.multicollinearity import check_multicollinearity


def test_single_feature_skips_check():
    X = pd.DataFrame({"x": np.random.normal(size=100)})
    y = 2 * X["x"] + np.random.normal(size=100)

    result = check_multicollinearity(X, y)
    assert result.passed
    assert "not applicable" in result.summary.lower()


def test_low_vif_passes():
    rng = np.random.default_rng(42)
    X = pd.DataFrame(
        {
            "x1": rng.normal(size=100),
            "x2": rng.normal(size=100),
        }
    )
    y = X["x1"] + X["x2"] + rng.normal(size=100)

    result = check_multicollinearity(X, y)
    assert result.passed
    for key in result.details:
        if "(VIF)" in key:
            assert result.details[key] < VIF_THRESHOLD


def test_high_vif_fails():
    rng = np.random.default_rng(42)
    x1 = rng.normal(size=100)
    x2 = x1 + rng.normal(0, 0.01, size=100)
    X = pd.DataFrame({"x1": x1, "x2": x2})
    y = 2 * x1 + 3 * x2 + rng.normal(size=100)

    result = check_multicollinearity(X, y)
    assert result.passed is False
    assert "fail" in result.summary.lower()


def test_severity_assignment():
    rng = np.random.default_rng(42)
    x1 = rng.normal(size=100)
    x2 = x1 + rng.normal(0, 0.01, size=100)
    X = pd.DataFrame({"x1": x1, "x2": x2})
    y = 2 * x1 + 3 * x2 + rng.normal(size=100)

    result = check_multicollinearity(X, y)
    assert result.severity == "high"


def test_detail_keys_are_present():
    rng = np.random.default_rng(42)
    x1 = rng.normal(size=100)
    x2 = rng.normal(size=100)
    X = pd.DataFrame({"x1": x1, "x2": x2})
    y = x1 + x2 + rng.normal(size=100)

    result = check_multicollinearity(X, y)
    for feature in X.columns:
        assert f"{feature} (VIF)" in result.details
        assert f"{feature} threshold" in result.details


def test_plot_is_returned():
    rng = np.random.default_rng(42)
    X = pd.DataFrame(
        {
            "x1": rng.normal(size=100),
            "x2": rng.normal(size=100),
        }
    )
    y = X["x1"] + X["x2"] + rng.normal(size=100)

    result = check_multicollinearity(X, y, return_plot=True)
    assert result.plot_base64 is not None
