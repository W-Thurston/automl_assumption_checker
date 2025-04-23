# tests/test_homoscedasticity.py
from src.core import homoscedasticity
from src.data import simulated_data
from src.models.linear_model_wrapper import LinearModelWrapper


def test_homoscedasticity_check_passes_on_linear_data():
    """
    Test that the Breusch-Pagan test passes on homoscedastic data.
    """
    df = simulated_data.generate_linear_data(n_samples=300, seed=42)
    result = homoscedasticity.check_homoscedasticity(df["x"], df["y"])
    assert result.passed
    assert result.details["breusch_pagan_pval"] > 0.05


def test_homoscedasticity_plot_generation():
    """
    Test that a base64-encoded plot is correctly generated and returned.
    """
    df = simulated_data.generate_linear_data(n_samples=300, seed=42)
    result = homoscedasticity.check_homoscedasticity(df["x"], df["y"], return_plot=True)
    assert result.plot_base64 is not None
    assert result.plot_base64.startswith("iVBOR")


def test_homoscedasticity_with_model_wrapper():
    """
    Ensure check_homoscedasticity works when a pre-fit model_wrapper is provided.
    """
    df = simulated_data.generate_linear_data(seed=123)
    wrapper = LinearModelWrapper(df["x"], df["y"]).fit()

    result = homoscedasticity.check_homoscedasticity(
        df["x"], df["y"], model_wrapper=wrapper
    )
    assert "breusch_pagan_pval" in result.details
