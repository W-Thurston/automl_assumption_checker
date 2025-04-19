# app/core/homoscedasticity.py
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan

from app.core.types import AssumptionResult
from app.utils import fig_to_base64

__all__ = ["check_homoscedasticity"]


def check_homoscedasticity(
    X: pd.Series, y: pd.Series, return_plot: bool = False
) -> AssumptionResult:
    """
    Check homoscedasticity using Breusch-Pagan test and residuals vs fitted plot.

    Args:
        X (pd.Series): Predictor (1D)
        y (pd.Series): Response (1D)
        return_plot (bool, optional): Whether to return base64-encoded
            PNG of the plot. Defaults to False.

    Returns:
        AssumptionResult: An object containing the outcome of the
            assumption check, including whether the check passed,
            a summary message, diagnostic details, optional residuals
            and fitted values, and an optional base64-encoded plot.
    """
    X_reshaped = X.values.reshape(-1, 1)
    model = sm.OLS(y, sm.add_constant(X_reshaped)).fit()
    residuals = model.resid
    fitted = model.fittedvalues

    # Breusch-Pagan test
    _, pval, _, _ = het_breuschpagan(residuals, sm.add_constant(X_reshaped))
    passed = pval > 0.05

    if return_plot:
        fig, ax = plt.subplots()
        ax.scatter(fitted, residuals, alpha=0.7)
        ax.axhline(0, color="red", linestyle="--")
        ax.set_xlabel("Fitted values")
        ax.set_ylabel("Residuals")
        ax.set_title("Residuals vs Fitted (Homoscedasticity Check)")
        encoded = fig_to_base64(fig)
    else:
        encoded = None

    return AssumptionResult(
        name="homoscedasticity",
        passed=passed,
        summary=f"Breusch-Pagan p = {pval:.4f} → {'Pass' if passed else 'Fail'}",
        details={"breusch_pagan_pval": pval},
        residuals=residuals,
        fitted=fitted,
        plot_base64=encoded,
    )
