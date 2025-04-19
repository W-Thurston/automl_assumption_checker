import pytest

from app.core import dispatcher
from app.data import simulated_data


def test_dispatch_single_assumption():
    df = simulated_data.generate_linear_data(seed=42)
    result = dispatcher.check_assumption("linearity", df["x"], df["y"])
    assert result.name == "linearity"
    assert result.passed


def test_dispatch_all_assumptions():
    df = simulated_data.generate_linear_data(seed=42)
    results = dispatcher.run_all_checks(df["x"], df["y"])
    assert "linearity" in results
    assert "homoskedasticity" in results
    assert results["linearity"].passed
    assert results["homoskedasticity"].passed


def test_unknown_assumption_raises():
    df = simulated_data.generate_linear_data(seed=42)
    with pytest.raises(ValueError):
        dispatcher.check_assumption("banana", df["x"], df["y"])
