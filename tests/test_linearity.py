# tests/test_linearity.py
import numpy as np
import pandas as pd

from app.core import linearity
from app.core.linearity import check_linearity
from app.data import simulated_data
from app.models.linear_model_wrapper import LinearModelWrapper


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


def test_linearity_with_model_wrapper():
    """
    Ensure check_linearity works when a pre-fit model_wrapper is provided.
    """
    X = pd.DataFrame({"x1": np.random.randn(100)})
    y = 2 * X["x1"] + np.random.randn(100)
    wrapper = LinearModelWrapper(X, y).fit()

    result = check_linearity(X, y, model_wrapper=wrapper)
    assert "r_squared" in result.details
