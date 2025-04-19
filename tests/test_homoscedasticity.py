# tests/test_homoscedasticity.py
from app.core import homoscedasticity
from app.data import simulated_data


def test_homoscedasticity_check_passes_on_linear_data():
    df = simulated_data.generate_linear_data(seed=42)
    result = homoscedasticity.check_homoscedasticity(df["x"], df["y"])
    assert result.passed
    assert result.details["breusch_pagan_pval"] > 0.05


def test_homoscedasticity_plot_generation():
    df = simulated_data.generate_linear_data(seed=42)
    result = homoscedasticity.check_homoscedasticity(df["x"], df["y"], return_plot=True)
    assert result.plot_base64 is not None
    assert result.plot_base64.startswith("iVBOR")
