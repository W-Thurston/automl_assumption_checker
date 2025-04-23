# tests/test_simulated_data.py
import numpy as np
import statsmodels.api as sm

from src.data import simulated_data


def test_generate_linear_data_shape():
    """
    Test that generated DataFrame has the correct shape and expected columns.
    """
    df = simulated_data.generate_linear_data(n_samples=150)
    assert df.shape == (150, 2)
    assert "x" in df.columns and "y" in df.columns


def test_generate_heteroscedastic_data_variance_growth():
    """
    Test that residual variance increases with X in heteroscedatic data.
    """
    df = simulated_data.generate_heteroscedastic_data(seed=123)
    # Rough proxy for growing variance
    var_low = df[df["x"] < 0]["y"].var()
    var_high = df[df["x"] > 0]["y"].var()
    assert var_high > var_low  # not guaranteed, but usually true


def test_heteroscedasticity_pattern_is_statistically_consistent():
    """
    Test that a linear model confirms increasing residual variance (heteroscedaticity).
    """
    df = simulated_data.generate_heteroscedastic_data(seed=123)
    X = sm.add_constant(np.abs(df["x"]))
    model = sm.OLS((df["y"] - 3 * df["x"]) ** 2, X).fit()
    coef = model.params.iloc[1]
    pval = model.pvalues.iloc[1]

    assert coef > 0, f"Expected positive slope for |x|, got {coef}"
    assert pval < 0.05, f"Expected significance, got p={pval}"


def test_generate_multicollinear_data_correlation():
    """
    Test that x1 and x2 are highly correlated in multicollinear data.
    """
    df = simulated_data.generate_multicollinear_data()
    corr = np.corrcoef(df["x1"], df["x2"])[0, 1]
    assert corr > 0.95


def test_generate_nonlinear_data_range():
    """
    Test that x values are within the expected range for nonlinear data.
    """
    df = simulated_data.generate_nonlinear_data()
    assert df["x"].between(-3, 3).all()
