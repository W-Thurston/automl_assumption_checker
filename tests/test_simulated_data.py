# tests/test_simulated_data.py
import numpy as np
import statsmodels.api as sm

from app.data import simulated_data


def test_generate_linear_data_shape():
    df = simulated_data.generate_linear_data(n_samples=150)
    assert df.shape == (150, 2)
    assert "x" in df.columns and "y" in df.columns


def test_generate_heteroskedastic_data_variance_growth():
    df = simulated_data.generate_heteroskedastic_data(seed=123)
    # Rough proxy for growing variance
    var_low = df[df["x"] < 0]["y"].var()
    var_high = df[df["x"] > 0]["y"].var()
    assert var_high > var_low  # not guaranteed, but usually true


def test_heteroskedasticity_pattern_is_statistically_consistent():
    df = simulated_data.generate_heteroskedastic_data(seed=123)
    X = sm.add_constant(np.abs(df["x"]))
    model = sm.OLS((df["y"] - 3 * df["x"]) ** 2, X).fit()
    coef = model.params.iloc[1]
    pval = model.pvalues.iloc[1]

    assert coef > 0, f"Expected positive slope for |x|, got {coef}"
    assert pval < 0.05, f"Expected significance, got p={pval}"


def test_generate_multicollinear_data_correlation():
    df = simulated_data.generate_multicollinear_data()
    corr = np.corrcoef(df["x1"], df["x2"])[0, 1]
    assert corr > 0.95


def test_generate_nonlinear_data_range():
    df = simulated_data.generate_nonlinear_data()
    assert df["x"].between(-3, 3).all()
