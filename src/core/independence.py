# src/core/independence.py
"""
Check for Independence of Residuals using
    - Plots:
        - Residuals vs Time (if time component exists)
        - Lag plot or autocorrelation function (ACF)
    - Statistical tests:
        - Durbin-Watson Test
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.stattools import durbin_watson

from src.config import INDEPENDENCE_DW_THRESHOLDS
from src.core.registry import register_assumption
from src.core.types import AssumptionResult
from src.utils import build_result, classify_severity, fig_to_base64

__all__ = ["check_independence"]


@register_assumption("independence", model_types=["linear"])
def check_independence(
    X: pd.Series, y: pd.Series, return_plot: bool = False, model_wrapper=None
) -> AssumptionResult:
    """

    This assumption ensures that residuals are not autocorrelated.
        Which is especially important in time series or ordered data.
        The Durbin-Watson statistic provides a formal test, and plots
        can visually highlight correlation patterns.

    Check for Independence of Residuals using
    - Plots:
        - Residuals vs Time (if time component exists)
        - Lag plot or autocorrelation function (ACF)
    - Statistical tests:
        - Durbin-Watson Test

    Args:
        X (pd.Series or pd.DataFrame): Predictor(s), 1D or multivariate
        y (pd.Series): Response variable
        return_plot (bool, optional): Whether to include diagnostic plots.
            Defaults to False.
        model_wrapper (optional): Shared fitted model object.
            If None, a linear model will be created.

    Returns:
        AssumptionResult: Structured diagnostic output.
    """

    # Guard for if model_wrapper is None
    if model_wrapper is None:
        from src.models.utils import get_model_wrapper

        model_wrapper = get_model_wrapper("linear", X, y)

    # Fit simple linear model to input data
    residuals = model_wrapper.residuals()
    y_pred = model_wrapper.fitted()

    # Durbin-Watson test
    dw = durbin_watson(resids=residuals)

    DW_LOWER_BOUND = INDEPENDENCE_DW_THRESHOLDS[0]
    DW_UPPER_BOUND = INDEPENDENCE_DW_THRESHOLDS[1]

    # Determine if test passed or failed based on thresholds
    passed = DW_LOWER_BOUND <= dw <= DW_UPPER_BOUND

    # Determine severity of violation based on Durbin-Watson value
    severity = classify_severity(dw, (DW_LOWER_BOUND, DW_UPPER_BOUND))

    # Recommend next steps if residuals are
    recommendation = (
        None
        if passed
        else (
            "Check for autocorrelation in residuals. "
            "Consider using time series models (e.g., ARIMA) or adding lag features."
        )
    )

    # Set flag for UI or prioritization
    flag = "info" if passed else "warning"

    # Plot Residual plot and Autocorrelation plot if requested
    plots = []
    if return_plot:
        # Residual Plot
        fig1, ax1 = plt.subplots()
        sns.residplot(x=y_pred, y=residuals, ax=ax1)
        ax1.set_title("Residuals vs Fitted")
        ax1.set_xlabel("Fitted Values")
        ax1.set_ylabel("Residuals")
        plots.append(
            {
                "title": "Residual Plot",
                "type": "residplot",
                "image": fig_to_base64(fig1),
            }
        )

        # ACF plot
        fig2, ax2 = plt.subplots()
        plot_acf(residuals, ax=ax2, lags=20)
        ax2.set_title("Autocorrelation Function (ACF) of Residuals")
        plots.append(
            {
                "title": "ACF",
                "type": "acf",
                "image": fig_to_base64(fig2),
            }
        )

    # Package the diagnostic results using the shared builder
    return build_result(
        name="independence",
        passed=passed,
        summary=f"Durbin-Watson statistic = {dw:.4f} → {'Pass' if passed else 'Fail'}",
        details={
            "durbin_watson": dw,
            "expected_range": f"{DW_LOWER_BOUND}-{DW_UPPER_BOUND}",
        },
        residuals=residuals,
        fitted=y_pred,
        plots=plots,
        severity=severity,
        recommendation=recommendation,
        flag=flag,
    )
