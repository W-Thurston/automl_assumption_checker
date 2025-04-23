# tests/test_independence.py

from src.core.independence import check_independence
from src.data import simulated_data
from src.models.linear_model_wrapper import LinearModelWrapper


def test_independence_passes_for_random_residuals():
    """
    Check that independence passes when residuals are i.i.d.
    """
    df = simulated_data.generate_linear_data(n_samples=200, seed=42)
    X = df[["x"]]
    y = df["y"]

    wrapper = LinearModelWrapper(X, y).fit()
    result = check_independence(X, y, model_wrapper=wrapper)

    assert result.passed
    assert "durbin_watson" in result.details


def test_independence_fails_for_autocorrelated_residuals():
    """
    Check that independence fails when residuals are autocorrelated.
    """
    df = simulated_data.generate_autocorrelated_data(n_samples=200, seed=42)
    X = df[["x"]]
    y = df["y"]

    wrapper = LinearModelWrapper(X, y).fit()
    result = check_independence(X, y, model_wrapper=wrapper)

    assert not result.passed
    assert "durbin_watson" in result.details
    assert (
        result.details["durbin_watson"] < 1.5 or result.details["durbin_watson"] > 2.5
    )


def test_independence_plot_generation():
    """
    Ensure check_independence returns plots if return_plot=True.
    """
    df = simulated_data.generate_linear_data(n_samples=200, seed=42)
    X = df[["x"]]
    y = df["y"]

    result = check_independence(X, y, return_plot=True)
    assert isinstance(result.plots, list)
    assert any("acf" in plot["title"].lower() for plot in result.plots)
