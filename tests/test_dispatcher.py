# tests/test_dispatcher.py
import pytest

from app.core import dispatcher
from app.data import simulated_data


def test_dispatch_single_assumption():
    """
    Test dispatcher's check_assumption().
    """
    df = simulated_data.generate_linear_data(seed=123)
    result = dispatcher.check_assumption("linearity", df["x"], df["y"])
    assert result.name == "linearity"
    assert result.passed


def test_dispatch_all_assumptions():
    """
    Test dispatcher's run_all_checks().
    """
    df = simulated_data.generate_linear_data(n_samples=300, seed=42)
    results, _ = dispatcher.run_all_checks(df["x"], df["y"], model_type="linear")
    assert "linearity" in results
    assert "homoscedasticity" in results
    assert results["linearity"].passed
    assert results["homoscedasticity"].passed


def test_unknown_assumption_raises():
    """
    Test an unknown assumption as input to check_assumption().
    """
    df = simulated_data.generate_linear_data(n_samples=300, seed=42)
    with pytest.raises(ValueError):
        dispatcher.check_assumption("banana", df["x"], df["y"])
