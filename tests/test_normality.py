from app.core import normality
from app.data import simulated_data
from app.models.linear_model_wrapper import LinearModelWrapper


def test_normality_check_passes_on_normal_data():
    """
    Test that normality check passes on simulated linear data with normal noise.
    """
    df = simulated_data.generate_linear_data(seed=42)
    result = normality.check_normality(df["x"], df["y"])
    assert result.passed
    assert result.details["shapiro_pval"] > 0.05
    assert result.details["dagostino_pval"] > 0.05


def test_normality_check_fails_on_nonnormal_data():
    """
    Test that normality check fails when residuals are clearly non-normal.
    """
    # Exaggerated sine wave to induce non-normal residuals
    df = simulated_data.generate_skewed_data(n_samples=100, seed=42)
    result = normality.check_normality(df["x"], df["y"])
    assert not result.passed


def test_normality_plot_generation():
    """
    Test that plot generation produces base64-encoded Q-Q and histogram plots.
    """
    df = simulated_data.generate_linear_data(seed=42)
    result = normality.check_normality(df["x"], df["y"], return_plot=True)
    assert result.plots is not None
    assert isinstance(result.plots, list)
    assert len(result.plots) >= 2
    assert all("image" in plot for plot in result.plots)
    assert all(plot["image"].startswith("iVBOR") for plot in result.plots)  # PNG base64


def test_normality_with_model_wrapper():
    """
    Ensure check_normality works when a pre-fit model_wrapper is provided.
    """
    df = simulated_data.generate_linear_data(seed=123)
    wrapper = LinearModelWrapper(df["x"], df["y"]).fit()

    result = normality.check_normality(df["x"], df["y"], model_wrapper=wrapper)
    assert "shapiro_pval" in result.details
