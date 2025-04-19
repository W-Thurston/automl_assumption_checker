# tests/test_linearity.py
from app.core import linearity
from app.data import simulated_data


def test_linearity_r_squared_threshold():
    """
    Test that the linearity assumption passes on clean linear data.
    """
    df = simulated_data.generate_linear_data(seed=123)
    result = linearity.check_linearity(df["x"], df["y"])
    assert result.details["r_squared"] > 0.7
    assert result.passed is True


def test_linearity_plot_generation():
    """
    Test that a base64-encoded plot is correctly generated and returned.
    """
    df = simulated_data.generate_linear_data(seed=123)
    result = linearity.check_linearity(df["x"], df["y"], return_plot=True)
    assert result.plot_base64 is not None
    assert result.plot_base64.startswith("iVBOR")  # PNG header in base64
